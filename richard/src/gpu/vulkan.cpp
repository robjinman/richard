#include "gpu/gpu.hpp"
#include "exception.hpp"
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <map>

#define VK_CHECK(fnCall, msg) \
  { \
    VkResult code = fnCall; \
    if (code != VK_SUCCESS) { \
      EXCEPTION(msg << " (result: " << code << ")"); \
    } \
  }

namespace {

const std::vector<const char*> ValidationLayers = {
  "VK_LAYER_KHRONOS_validation"
};

class Vulkan : public Gpu {
  public:
    Vulkan();

    ShaderHandle compileShader(const std::string& shaderSource);
    void submitBuffer(const void* buffer, size_t bufferSize) override;
    void executeShader(size_t shaderIndex, size_t numWorkgroups) override;
    void retrieveBuffer(void* data) override;

    ~Vulkan();

  private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT,
      VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void*);

    void checkValidationLayerSupport() const;
#ifndef NDEBUG
    VkDebugUtilsMessengerCreateInfoEXT getDebugMessengerCreateInfo() const;
#endif
    std::vector<const char*> getRequiredExtensions() const;
    void createVulkanInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    uint32_t findComputeQueueFamily() const;
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
      VkBuffer& buffer, VkDeviceMemory& bufferMemory) const;
    void createDescriptorSetLayout();
    void createPipelineLayout();
    void createCommandPool();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffer();
    void recordCommandBuffer(VkCommandBuffer commandBuffer, size_t numWorkgroups);
    void createSyncObjects();
#ifndef NDEBUG
    void destroyDebugMessenger();
#endif
    void destroyBuffer();
    void destroyStagingBuffer();
    VkShaderModule createShaderModule(const std::string& source) const;
    inline VkPipeline currentPipeline() const;

    VkInstance m_instance;
#ifndef NDEBUG
    VkDebugUtilsMessengerEXT m_debugMessenger;
#endif
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VkQueue m_computeQueue;
    VkBuffer m_buffer;
    VkDeviceMemory m_bufferMemory;
    VkDeviceSize m_bufferSize;
    VkBuffer m_stagingBuffer;
    VkDeviceMemory m_stagingBufferMemory;
    VkDescriptorSetLayout m_descriptorSetLayout;
    VkPipelineLayout m_pipelineLayout;
    std::vector<VkPipeline> m_pipelines;
    size_t m_currentPipelineIdx;
    VkCommandPool m_commandPool;
    VkCommandBuffer m_commandBuffer;
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSet m_descriptorSet;
    VkFence m_taskCompleteFence;
};

Vulkan::Vulkan()
  : m_buffer(VK_NULL_HANDLE)
  , m_bufferMemory(VK_NULL_HANDLE)
  , m_bufferSize(0)
  , m_stagingBuffer(VK_NULL_HANDLE)
  , m_stagingBufferMemory(VK_NULL_HANDLE) {

  createVulkanInstance();
#ifndef NDEBUG
  setupDebugMessenger();
#endif
  pickPhysicalDevice();
  createLogicalDevice();
  createDescriptorSetLayout();
  createPipelineLayout();
  createCommandPool();
  createDescriptorPool();
  createCommandBuffer();
  createSyncObjects();
}

void Vulkan::destroyBuffer() {
    vkDestroyBuffer(m_device, m_buffer, nullptr);
    vkFreeMemory(m_device, m_bufferMemory, nullptr);
    m_buffer = VK_NULL_HANDLE;
    m_bufferMemory = VK_NULL_HANDLE;
    m_bufferSize = 0;
}

void Vulkan::destroyStagingBuffer() {
  vkDestroyBuffer(m_device, m_stagingBuffer, nullptr);
  vkFreeMemory(m_device, m_stagingBufferMemory, nullptr);
  m_stagingBuffer = VK_NULL_HANDLE;
  m_stagingBufferMemory = VK_NULL_HANDLE;
}

void Vulkan::submitBuffer(const void* data, size_t size) {
  VK_CHECK(vkDeviceWaitIdle(m_device), "Error waiting for device to be idle");

  if (m_buffer != VK_NULL_HANDLE) {
    destroyBuffer();
  }

  if (m_stagingBuffer != VK_NULL_HANDLE) {
    destroyStagingBuffer();
  }

  VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                              | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                              | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

  createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, flags, m_stagingBuffer,
    m_stagingBufferMemory);

  void* stagingBufferMapped = nullptr;
  vkMapMemory(m_device, m_stagingBufferMemory, 0, size, 0, &stagingBufferMapped);
  memcpy(stagingBufferMapped, data, size);
  vkUnmapMemory(m_device, m_stagingBufferMemory);

  VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
                           | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                           | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  createBuffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_buffer, m_bufferMemory);

  copyBuffer(m_stagingBuffer, m_buffer, size);

  m_bufferSize = size;

  createDescriptorSets();
}

ShaderHandle Vulkan::compileShader(const std::string& shaderSource) {
  VkShaderModule shaderModule = createShaderModule(shaderSource);

  VkPipelineShaderStageCreateInfo shaderStageInfo{};
  shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageInfo.module = shaderModule;
  shaderStageInfo.pName = "main";

  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.layout = m_pipelineLayout;
  pipelineInfo.stage = shaderStageInfo;

  VkPipeline pipeline = VK_NULL_HANDLE;

  VK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
    &pipeline), "Failed to create compute pipeline");

  vkDestroyShaderModule(m_device, shaderModule, nullptr);

  m_pipelines.push_back(pipeline);

  return m_pipelines.size() - 1;
}

void Vulkan::executeShader(size_t shaderIndex, size_t numWorkgroups) {
  //VK_CHECK(vkDeviceWaitIdle(m_device), "Error waiting for device to be idle");

  m_currentPipelineIdx = shaderIndex;

  vkResetCommandBuffer(m_commandBuffer, 0);
  recordCommandBuffer(m_commandBuffer, numWorkgroups);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &m_commandBuffer;

  VK_CHECK(vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_taskCompleteFence),
    "Failed to submit compute command buffer");

  VK_CHECK(vkWaitForFences(m_device, 1, &m_taskCompleteFence, VK_TRUE, UINT64_MAX),
    "Error waiting for fence");

  VK_CHECK(vkResetFences(m_device, 1, &m_taskCompleteFence), "Error resetting fence");
}

void Vulkan::retrieveBuffer(void* data) {
//  VK_CHECK(vkDeviceWaitIdle(m_device), "Error waiting for device to be idle");

  if (m_buffer == VK_NULL_HANDLE) {
    EXCEPTION("Error retrieving buffer; Buffer has not been created yet");
  }

  DBG_ASSERT(m_stagingBuffer != VK_NULL_HANDLE);

  copyBuffer(m_buffer, m_stagingBuffer, m_bufferSize);

  void* stagingBufferMapped = nullptr;
  vkMapMemory(m_device, m_stagingBufferMemory, 0, m_bufferSize, 0, &stagingBufferMapped);
  memcpy(data, stagingBufferMapped, m_bufferSize);
  vkUnmapMemory(m_device, m_stagingBufferMemory);
}

VkPipeline Vulkan::currentPipeline() const {
  return m_pipelines.at(m_currentPipelineIdx);
}

#ifndef NDEBUG
void Vulkan::checkValidationLayerSupport() const {
  uint32_t layerCount;
  VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, nullptr),
    "Failed to enumerate instance layer properties");

  std::vector<VkLayerProperties> available(layerCount);
  VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, available.data()),
    "Failed to enumerate instance layer properties");

  for (auto layer : ValidationLayers) {
    auto fnMatches = [=](const VkLayerProperties& p) {
      return strcmp(layer, p.layerName) == 0;
    };
    if (std::find_if(available.begin(), available.end(), fnMatches) == available.end()) {
      EXCEPTION("Validation layer '" << layer << "' not supported");
    }
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL Vulkan::debugCallback(
  VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
  const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {

  std::cerr << "Validation layer: " << data->pMessage << std::endl;

  return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT Vulkan::getDebugMessengerCreateInfo() const {
  VkDebugUtilsMessengerCreateInfoEXT createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                             | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                             | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                         | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                         | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
  createInfo.pUserData = nullptr;
  return createInfo;
}

void Vulkan::setupDebugMessenger() {
  auto createInfo = getDebugMessengerCreateInfo();

  auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
    vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
  if (func == nullptr) {
    EXCEPTION("Error getting pointer to vkCreateDebugUtilsMessengerEXT()");
  }
  VK_CHECK(func(m_instance, &createInfo, nullptr, &m_debugMessenger),
    "Error setting up debug messenger");
}

void Vulkan::destroyDebugMessenger() {
  auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
    vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
  func(m_instance, m_debugMessenger, nullptr);
}
#endif

std::vector<const char*> Vulkan::getRequiredExtensions() const {
  std::vector<const char*> extensions;

#ifndef NDEBUG
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

  return extensions;
}

void Vulkan::pickPhysicalDevice() {
  uint32_t deviceCount = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr),
    "Failed to enumerate physical devices");

  if (deviceCount == 0) {
    EXCEPTION("No physical devices found");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data()),
    "Failed to enumerate physical devices");

  m_physicalDevice = devices[0];
}

uint32_t Vulkan::findComputeQueueFamily() const {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount,
    queueFamilies.data());

  for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      return i;
    }
  }

  EXCEPTION("Could not find compute queue family");
}

void Vulkan::createLogicalDevice() {
  VkDeviceQueueCreateInfo queueCreateInfo{};

  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = findComputeQueueFamily();
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkPhysicalDeviceFeatures deviceFeatures{};

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.queueCreateInfoCount = 1;
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.pEnabledFeatures = &deviceFeatures;
  createInfo.enabledExtensionCount = 0;

#ifdef NDEBUG
  createInfo.enabledLayerCount = 0;
#else
  createInfo.enabledLayerCount = ValidationLayers.size();
  createInfo.ppEnabledLayerNames = ValidationLayers.data();
#endif

  VK_CHECK(vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device),
    "Failed to create logical device");

  vkGetDeviceQueue(m_device, queueCreateInfo.queueFamilyIndex, 0, &m_computeQueue);
}

void Vulkan::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = m_commandPool; // TODO: Separate pool for temp buffers?
  allocInfo.commandBufferCount = 1;
  
  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);
  
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);
  
  VkBufferCopy copyRegion{};
  copyRegion.srcOffset = 0;
  copyRegion.dstOffset = 0;
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
  
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  
  vkQueueSubmit(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(m_computeQueue); // Use fence if doing multiple transfers simultaneously
  
  vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
}

void Vulkan::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
  VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) const {

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  bufferInfo.flags = 0;

  VK_CHECK(vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer), "Failed to create buffer");

  auto findMemoryType = [this, properties](uint32_t typeFilter) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
      if (typeFilter & (1 << i) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {

        return i;
      }
    }

    EXCEPTION("Failed to find suitable memory type");
  };

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits);

  VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory),
    "Failed to allocate memory for buffer");

  vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
}

void Vulkan::createVulkanInstance() {
#ifndef NDEBUG
  checkValidationLayerSupport();
#endif

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Vulkan Compute Examples";
  appInfo.applicationVersion = VK_MAKE_API_VERSION(1, 0, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_API_VERSION(1, 0, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;
#ifdef NDEBUG
  createInfo.enabledLayerCount = 0;
  createInfo.pNext = nullptr;
#else
  createInfo.enabledLayerCount = ValidationLayers.size();
  createInfo.ppEnabledLayerNames = ValidationLayers.data();

  auto debugMessengerInfo = getDebugMessengerCreateInfo();
  createInfo.pNext = &debugMessengerInfo;
#endif

  auto extensions = getRequiredExtensions();

  createInfo.enabledExtensionCount = extensions.size();
  createInfo.ppEnabledExtensionNames = extensions.data();

  VK_CHECK(vkCreateInstance(&createInfo, nullptr, &m_instance), "Failed to create instance");
}

void Vulkan::createCommandPool() {
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = findComputeQueueFamily();
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  VK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool),
    "Failed to create command pool");
}

void Vulkan::createCommandBuffer() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = m_commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &m_commandBuffer),
    "Failed to allocate command buffer");
}

VkShaderModule Vulkan::createShaderModule(const std::string& source) const {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;
  auto result = compiler.CompileGlslToSpv(source, shaderc_shader_kind::shaderc_glsl_compute_shader,
    "shader", options);

  if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
    EXCEPTION("Error compiling shader: " << result.GetErrorMessage());
  }

  std::vector<uint32_t> code;
  code.assign(result.cbegin(), result.cend());

  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size() * sizeof(uint32_t);
  createInfo.pCode = code.data();

  VkShaderModule shaderModule;
  VK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule),
    "Failed to create shader module");

  return shaderModule;
}

void Vulkan::createDescriptorSetLayout() {
  VkDescriptorSetLayoutBinding layoutBinding{};
  layoutBinding.binding = 0;
  layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  layoutBinding.descriptorCount = 1;
  layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  layoutBinding.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &layoutBinding;
  
  VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout),
    "Failed to create descriptor set layout");
}

void Vulkan::createDescriptorPool() {
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 1;

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = 1;

  VK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool),
    "Failed to create descriptor pool");
}

void Vulkan::createDescriptorSets() {
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = m_descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &m_descriptorSetLayout;

  VK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet),
    "Failed to allocate descriptor set");

  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = m_buffer;
  bufferInfo.offset = 0;
  bufferInfo.range = m_bufferSize;

  VkWriteDescriptorSet descriptorWrite{};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = m_descriptorSet;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pBufferInfo = &bufferInfo;
  descriptorWrite.pImageInfo = nullptr;
  descriptorWrite.pTexelBufferView = nullptr;

  vkUpdateDescriptorSets(m_device, 1, &descriptorWrite, 0, nullptr);
}

void Vulkan::createPipelineLayout() { 
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 0;
  pipelineLayoutInfo.pPushConstantRanges = nullptr;
  VK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout),
    "Failed to create pipeline layout");
}

void Vulkan::recordCommandBuffer(VkCommandBuffer commandBuffer, size_t numWorkgroups) {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = 0;
  beginInfo.pInheritanceInfo = nullptr;

  VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo),
    "Failed to begin recording command buffer");

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, currentPipeline());
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1,
    &m_descriptorSet, 0, 0);
  vkCmdDispatch(commandBuffer, numWorkgroups, 1, 1);

  VK_CHECK(vkEndCommandBuffer(commandBuffer), "Failed to record command buffer");
}

void Vulkan::createSyncObjects() {
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = 0;

  VK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &m_taskCompleteFence),
    "Failed to create fence");
}

Vulkan::~Vulkan() {
  vkDestroyFence(m_device, m_taskCompleteFence, nullptr);
  vkDestroyCommandPool(m_device, m_commandPool, nullptr);
  for (VkPipeline pipeline : m_pipelines) {
    vkDestroyPipeline(m_device, pipeline, nullptr);
  }
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  destroyBuffer();
  destroyStagingBuffer();
  vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
#ifndef NDEBUG
  destroyDebugMessenger();
#endif
  vkDestroyDevice(m_device, nullptr);
  vkDestroyInstance(m_instance, nullptr);
}

}

GpuPtr createGpu() {
  return std::make_unique<Vulkan>();
}
