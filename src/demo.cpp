#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <string>

#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xaxis_iterator.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "matplot/matplot.h"

// TODO: not exactly the cleanest fn signature
std::tuple<xt::xarray<double>,xt::xarray<double>,xt::xarray<double>,xt::xarray<double>>
readData(std::string observation_file, std::string group_file)
{
    std::ifstream in_file;
    in_file.open(observation_file);
    xt::xarray<double> observations = xt::load_csv<double>(in_file);
    in_file.close();

    in_file.open(group_file);
    xt::xarray<double> groups = xt::load_csv<double>(in_file);
    in_file.close();

    xt::xarray<double> combined = xt::hstack(xt::xtuple(observations, groups));

    xt::random::shuffle(combined);

    int training_rows = combined.shape()[0] * 0.8;
    int obs_count = combined.shape()[1] - 1;

    auto training_full = xt::view(combined, xt::range(0, training_rows), xt::all());
    auto testing_full = xt::view(combined, xt::range(training_rows, -1), xt::all());

    auto training_obs = xt::view(training_full, xt::all(), xt::range(0, obs_count));
    auto training_grp = xt::view(training_full, xt::all(), xt::range(obs_count, obs_count + 1));

    auto testing_obs = xt::view(testing_full, xt::all(), xt::range(0, obs_count));
    auto testing_grp = xt::view(testing_full, xt::all(), xt::range(obs_count, obs_count + 1));

    return {training_obs, training_grp, testing_obs, testing_grp};
}

void plotPCA(xt::xarray<double> observations, xt::xarray<double> groups, xt::xarray<double> Vxyz)
{
    xt::xarray<double> Vx = xt::view(Vxyz, xt::range(0, 1), xt::all());
    xt::xarray<double> Vy = xt::view(Vxyz, xt::range(1, 2), xt::all());
    xt::xarray<double> Vz = xt::view(Vxyz, xt::range(2, 3), xt::all());

    // Prepare data for plotting
    size_t sample_cnt = observations.shape()[0];
    std::vector<double> colors(groups.begin(), groups.end());
    //std::transform(
    //    colors.cbegin(),
    //    colors.cend(),
    //    colors.begin(),
    //    [](auto x) { return x ? 2 : 1; }
    //);

    std::vector<double> xs, ys, zs;
    xs.reserve(sample_cnt);
    ys.reserve(sample_cnt);
    zs.reserve(sample_cnt);

    auto it = xt::axis_begin(observations, 0);
    auto end = xt::axis_end(observations, 0);

    while (it != end) {
        xs.push_back(xt::linalg::dot(Vx, *it)(0));
        ys.push_back(xt::linalg::dot(Vy, *it)(0));
        zs.push_back(xt::linalg::dot(Vz, *it)(0));
        ++it;
    }

    matplot::scatter3(xs, ys, zs, std::vector<double>{}, colors)->marker_face(true);
    matplot::show();
}

int main()
{
    xt::random::seed(80085);

    auto [train_obs, train_grp, test_obs, test_grp] =
        readData("data/ovariancancer_obs.csv", "data/ovariancancer_grp.csv");

    // Calculate SVD
    auto [U, S, VT] = xt::linalg::svd(train_obs, false);

    plotPCA(train_obs, train_grp, xt::view(VT, xt::range(0, 3), xt::all()));

    return 0;
}
