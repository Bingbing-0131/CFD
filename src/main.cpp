#include <iostream>
#include <iomanip>  // for std::setprecision
#include <Eigen/Dense>
#include "upwind.h"  // Include your header
#include "Implicit_Euler.h"

int main()
{
    double a = 1.0;
    double CFL = 0.5;
    int mx = 100;
    int num_step;

    ImplicitEuler compute;
    std::string folder = "../Implicit_Euler";

    compute.Initialize(mx, CFL, a, folder);
    std::cout << "Time step dt = " << std::setprecision(6) << compute.dt_() << std::endl;

    num_step = static_cast<int>(20.0 / compute.dt_());

    compute.Data(0);

    for (int i = 0; i < num_step; i++)
    {
        compute.Forward();
        double time = i * compute.dt_();
        std::cout<<i<<std::endl;
        // Save at t = 0.1, 1.0, 10.0 using small epsilon to handle float comparison
        if (std::abs(time - 0.1) < 1e-6 ||
            std::abs(time - 1.0) < 1e-6 ||
            std::abs(time - 10.0) < 1e-6)
        {
            compute.Data(i);  // Optionally pass filename or label
            std::cout << "Saved data at t = " << time << std::endl;
        }
    }

    return 0;
}
