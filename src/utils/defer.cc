#include "auto_engine/utils/defer.h"
#include <functional>

namespace utils {

Defer::Defer(std::function<void()> f): _f(f) {}

Defer::~Defer() {
    _f();
}

}