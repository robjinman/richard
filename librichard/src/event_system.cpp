#include "richard/event_system.hpp"
#include <map>

namespace richard {
namespace {

class EventSystemImpl : public EventSystem {
  friend class ::richard::EventHandle;

  public:
    EventHandle listen(eventId_t eventId, EventHandler handler) override;
    void raise(const Event& event) override;

  private:
    static handlerId_t nextId;
    std::map<eventId_t, std::map<handlerId_t, EventHandler>> m_handlers;
};

handlerId_t EventSystemImpl::nextId = 1;

EventHandle EventSystemImpl::listen(eventId_t eventId, EventHandler handler) {
  handlerId_t handlerId = nextId++;
  m_handlers[eventId].insert(std::make_pair(handlerId, handler));
  return EventHandle{*this, eventId, handlerId};
}

void EventSystemImpl::raise(const Event& event) {
  const auto& handlers = m_handlers[event.id()];
  for (const auto& i : handlers) {
    i.second(event);
  }
}

}

EventHandle::EventHandle(EventSystem& eventSystem, eventId_t eventId, handlerId_t handlerId)
  : m_eventSystem(dynamic_cast<EventSystemImpl&>(eventSystem))
  , m_eventId(eventId)
  , m_handlerId(handlerId) {}

EventHandle::EventHandle(EventHandle&& mv)
  : m_eventSystem(mv.m_eventSystem)
  , m_eventId(mv.m_eventId)
  , m_handlerId(mv.m_handlerId) {

  mv.m_handlerId = 0;
}

EventHandle::~EventHandle() {
  if (m_handlerId != 0) {
    dynamic_cast<EventSystemImpl&>(m_eventSystem).m_handlers[m_eventId].erase(m_handlerId);
  }
}

EventSystemPtr createEventSystem() {
  return std::make_unique<EventSystemImpl>();
}

}
