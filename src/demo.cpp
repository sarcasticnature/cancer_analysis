#include <iostream>
#include <fstream>
#include <tuple>

#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor-blas/xlinalg.hpp"

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
    int observation_cnt = combined.shape()[1];

    auto training_full = xt::view(combined, xt::range(0, training_rows), xt::all());
    auto testing_full = xt::view(combined, xt::range(training_rows, -1), xt::all());

    auto training_obs = xt::view(training_full, xt::all(), xt::range(0, observation_cnt));
    auto training_grp = xt::view(training_full, xt::all(), xt::range(observation_cnt, -1));

    auto testing_obs = xt::view(testing_full, xt::all(), xt::range(0, observation_cnt));
    auto testing_grp = xt::view(testing_full, xt::all(), xt::range(observation_cnt, -1));

    return {training_obs, training_grp, testing_obs, testing_grp};
}


int main()
{
    xt::random::seed(80085);

    auto [train_obs, train_grp, test_obs, test_grp] =
        readData("data/ovariancancer_obs.csv", "data/ovariancancer_grp.csv");

    std::cout << "Hello Cancer" << std::endl;

    return 0;
}
