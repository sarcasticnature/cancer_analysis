#include <fstream>
#include <tuple>
#include <vector>
#include <string>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xaxis_iterator.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "matplot/matplot.h"

namespace cancer_analysis
{

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

void plotData(xt::xarray<double> observations, xt::xarray<double> groups, xt::xarray<double> VTxyz)
{
    xt::xarray<double> Vx = xt::view(VTxyz, xt::range(0, 1), xt::all());
    xt::xarray<double> Vy = xt::view(VTxyz, xt::range(1, 2), xt::all());
    xt::xarray<double> Vz = xt::view(VTxyz, xt::range(2, 3), xt::all());

    size_t sample_cnt = observations.shape()[0];
    std::vector<double> colors(groups.begin(), groups.end());

    std::vector<double> xs, ys, zs;
    xs.reserve(sample_cnt);
    ys.reserve(sample_cnt);
    zs.reserve(sample_cnt);

    // Plotting twice because matplot is dumb (and has ugly default colors)
    // I haven't found a different way to change the colors that actually works

    xt::xarray<double> combined = xt::hstack(xt::xtuple(observations, groups));
    size_t grp_idx = combined.shape()[1] - 1;
    auto it = xt::axis_begin(combined, 0);
    auto end = xt::axis_end(combined, 0);

    while (it != end) {
        if (!it->at(grp_idx)) {
            using namespace xt::placeholders;
            auto obs = xt::view(*it, xt::range(_, -1));
            xs.push_back(xt::linalg::dot(Vx, obs)(0));
            ys.push_back(xt::linalg::dot(Vy, obs)(0));
            zs.push_back(xt::linalg::dot(Vz, obs)(0));
        }
        ++it;
    }

    matplot::figure();
    matplot::hold(matplot::on);
    matplot::scatter3(xs, ys, zs)->marker_face(true).marker_size(10);
    xs.clear();
    ys.clear();
    zs.clear();

    // can't reassign iterators, so dumb
    auto it_ = xt::axis_begin(combined, 0);

    while (it_ != end) {
        if (it_->at(grp_idx)) {
            using namespace xt::placeholders;
            auto obs = xt::view(*it_, xt::range(_, -1));
            xs.push_back(xt::linalg::dot(Vx, obs)(0));
            ys.push_back(xt::linalg::dot(Vy, obs)(0));
            zs.push_back(xt::linalg::dot(Vz, obs)(0));
        }
        ++it_;
    }

    matplot::scatter3(xs, ys, zs)->marker_face(true).marker_size(10);
    matplot::hold(matplot::off);
    matplot::show();
}

template<typename T>
xt::xarray<T> normalizeData(xt::xarray<T> data)
{
    // TODO: the xt::stddev call is slow... consider replacing?
    // TODO: remove print statements after replacement
    std::cout << "TODO: xt::stddev is slow, consider replacing" << std::endl;
    xt::xarray<T> res = (data - xt::mean(data, 0)) / xt::stddev(data, {0});
    std::cout << "normalization finished" << std::endl;
    return res;
}

} // namespace cancer_analysis

