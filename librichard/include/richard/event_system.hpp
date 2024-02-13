#pragma once

#include "utils.hpp"
#include <memory>
#include <functional>

namespace richard {

using eventId_t = hashedString_t;

class Event {
  public:
    Event(eventId_t id)
      : m_id(id) {}

    eventId_t id() const {
      return m_id;
    }

    virtual ~Event() {}

  private:
    eventId_t m_id;
};

using EventHandler = std::function<void(const Event& event)>;
using handlerId_t = long;

class EventSystem;

class EventHandle {
  public:
    EventHandle(EventSystem& eventSystem, eventId_t eventId, handlerId_t handlerId);
    EventHandle(const EventHandle& cpy) = delete;
    EventHandle(EventHandle&& mv);

    ~EventHandle();

  private:
    EventSystem& m_eventSystem;
    eventId_t m_eventId;
    handlerId_t m_handlerId;
};

class EventSystem {
  public:
    virtual EventHandle listen(eventId_t eventId, EventHandler handler) = 0;
    virtual void raise(const Event& event) = 0;

    virtual ~EventSystem() {}
};

using EventSystemPtr = std::unique_ptr<EventSystem>;

EventSystemPtr createEventSystem();

}
