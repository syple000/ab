#ifndef UTILS_DEFER_H
#define UTILS_DEFER_H

#include <functional>

namespace utils {

class Defer {
public:
    Defer(std::function<void()>);
    ~Defer();
private:
    std::function<void()> _f;
};

}

#endif 