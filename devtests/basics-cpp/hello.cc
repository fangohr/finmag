//How did this C++ thing work? Basic examples.


#include <iostream>
#include <string>
#include <sstream>
std::string hello(std::string name) {
            std::stringstream ss;
            ss << "Hello" << name << std::endl;
            return ss.str();
        }


int main(void) {
	std::cout << hello(" test");
	return 0;
}


