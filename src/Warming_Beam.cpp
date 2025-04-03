#include "Warming_Beam.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

void WarmingBeam::Initialize(int mx_, long double CFL_, long double a_, std::string folder_) {
    mx = mx_;
    dx = 1.0L / mx;
    CFL = CFL_;
    a = a_;
    dt = CFL * dx / a;

    u_prev.resize(mx + 1);
    u_new.resize(mx + 1);
    
    for (int i = 0; i <= mx; i++) {
        long double x = i * dx;
        u_prev[i] = Initial_condition(x);
        u_new[i] = u_prev[i];
    }

    folder = folder_;
}

long double WarmingBeam::Initial_condition(long double x) {
    x -= 0.5L;  // Center the domain at 0

    if (x >= -0.25L && x <= 0.25L) return 1.0L;
    return 0.0L;
}

void WarmingBeam::Forward() {
    std::vector<long double> u_temp(mx + 1);

    for (int i = 1; i <= mx; i++) {
        // Circular shift implementation
        int i_minus_1 = (i - 1 + mx + 1) % (mx + 1);
        int i_minus_2 = (i - 2 + mx + 1) % (mx + 1);

        u_temp[i] = u_prev[i]
            - 0.5L * CFL * (3.0L * u_prev[i] - 4.0L * u_prev[i_minus_1] + u_prev[i_minus_2])
            + 0.5L * CFL * CFL * (u_prev[i] - 2.0L * u_prev[i_minus_1] + u_prev[i_minus_2]);
    }
    u_temp[0] = u_new[mx];
    u_prev = u_temp;
    u_new = u_temp;
}

void WarmingBeam::Data(int step) {
    std::ostringstream filename;
    filename << folder << "/output_step_" << step << ".txt";

    std::ofstream file(filename.str());
    // Use std::scientific to preserve precision
    file << std::scientific;
    for (int i = 0; i <= mx; i++) {
        file << i * dx << " " << u_new[i] << "\n";
    }
    file.close();
}