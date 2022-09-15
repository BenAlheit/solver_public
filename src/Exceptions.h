#ifndef SOLVER_EXCEPTIONS_H
#define SOLVER_EXCEPTIONS_H

#include <string>

using namespace std;

class NotImplemented : public logic_error
{
public:
    NotImplemented(string message) : logic_error(message) { };
};

#endif //SOLVER_EXCEPTIONS_H
