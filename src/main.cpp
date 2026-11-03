#include "main.hpp"

#include <iostream>

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
    std::cout << "Hello, world!" << std::endl;






    }
    Kokkos::finalize();
    return 0;
}
