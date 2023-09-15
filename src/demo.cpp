#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <string>

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
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

void plotData(xt::xarray<double> observations, xt::xarray<double> groups, xt::xarray<double> VTxyz)
{
    xt::xarray<double> Vx = xt::view(VTxyz, xt::range(0, 1), xt::all());
    xt::xarray<double> Vy = xt::view(VTxyz, xt::range(1, 2), xt::all());
    xt::xarray<double> Vz = xt::view(VTxyz, xt::range(2, 3), xt::all());

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


std::tuple<xt::xarray<double>, xt::xtensor_fixed<double, xt::xshape<2, 3>>>
computeKMeans(xt::xarray<double> observations,
              xt::xarray<double> VTxyz,
              unsigned max_iters)
{
    xt::xarray<double> points = xt::linalg::dot(observations, xt::transpose(VTxyz));

    size_t point_count = points.shape()[0];
    xt::xarray<double> groups = xt::ones<double>(std::vector<size_t>{point_count, 1});
    xt::xarray<double> old_groups = groups;

    xt::xarray<bool> mask;
    xt::xarray<double> tmp; // TODO: better name
    xt::xtensor_fixed<double, xt::xshape<1, 3>> k0, k1, p;
    // initialize cluster centers -- TODO: randomize
    k0 = xt::view(points, xt::range(2, 3), xt::all());
    k1 = xt::view(points, xt::range(8, 9), xt::all());

    for (unsigned i = 0; i < max_iters; ++i) {
        for (size_t j = 0; j < point_count; ++j) {
            p = xt::view(points, xt::range(j, j + 1), xt::all());
            //std::cout << "p=" << p << std::endl;
            //std::cout << "k1=" << k1 << std::endl;
            //std::cout << "k1-p=" << (k1 - p) << std::endl;
            //std::cout << "norm k0 - p=" << xt::linalg::norm(xt::eval(k0 - p)) << std::endl;
            //std::cout << "norm k1 - p=" << xt::linalg::norm(xt::eval(k1 - p)) << std::endl;
            //std::cout << (xt::linalg::norm(k0 - p) - xt::linalg::norm(k1 - p)) << std::endl;
            if (xt::linalg::norm(k0 - p) < xt::linalg::norm(k1 - p)) {
                groups(j, 0) = 0.0;
            } else {
                groups(j, 0) = 1.0;
            }
        }

        if (groups == old_groups) break;
        old_groups = groups;

        // TODO: difficult to parse, split into its own fn (lambda?)?
        //k0 = xt::mean(xt::view(points, xt::drop(xt::argwhere(groups > 0.5)), xt::all()), 0);
        //k1 = xt::mean(xt::view(points, xt::keep(xt::argwhere(groups > 0.5)), xt::all()), 0);
        mask = groups > 0.5;
        mask.reshape({-1, 1});
        tmp = xt::filter(points, xt::hstack(xt::xtuple(mask, mask, mask)));
        k0 = xt::mean(tmp.reshape({-1, 3}), 0);
        mask = groups < 0.5;
        mask.reshape({-1, 1});
        //std::cout << xt::hstack(xt::xtuple(mask, groups)) << std::endl;
        tmp = xt::filter(points, xt::hstack(xt::xtuple(mask, mask, mask)));
        k1 = xt::mean(tmp.reshape({-1, 3}), 0);
    }

    auto centers = xt::concatenate(xt::xtuple(k0, k1), 0);
    return {groups, centers};
}

xt::xarray<double> testKMeans(xt::xarray<double> observations,
                              xt::xarray<double> VTxyz,
                              xt::xtensor_fixed<double, xt::xshape<2, 3>> centers)
{
    xt::xarray<double> points = xt::linalg::dot(observations, xt::transpose(VTxyz));

    size_t point_count = points.shape()[0];
    xt::xarray<double> groups = xt::empty<double>(std::vector<size_t>{point_count, 1});

    xt::xtensor_fixed<double, xt::xshape<1, 3>> k0, k1, p;
    k0 = xt::view(centers, xt::range(0, 1), xt::all());
    k1 = xt::view(centers, xt::range(1, 2), xt::all());

    
    for (size_t i = 0; i < point_count; ++i) {
        p = xt::view(points, xt::range(i, i + 1), xt::all());
        if (xt::linalg::norm(k0 - p) < xt::linalg::norm(k1 - p)) {
            groups(i, 0) = 0.0;
        } else {
            groups(i, 0) = 1.0;
        }
    }

    return groups;
}

int main()
{
    xt::random::seed(80085);
    //xt::random::seed(1337);

    auto [train_obs, train_grp, test_obs, test_grp] =
        readData("data/ovariancancer_obs.csv", "data/ovariancancer_grp.csv");

    // Normalize the data and calculate the SVD
    // TODO: the xt::stddev call is slow... consider replacing?
    std::cout << "TODO: xt::stddev is slow, consider replacing" << std::endl;
    xt::xarray<double> X = (train_obs - xt::mean(train_obs, 0)) / xt::stddev(train_obs, {0});
    std::cout << "xt::stddev finished" << std::endl;

    auto [U, S, VT] = xt::linalg::svd(X, false);
    // Use the first three principal components
    auto VTxyz = xt::view(VT, xt::range(0, 3), xt::all());
    plotData(X, train_grp, VTxyz);

    auto [kmeans_training_grp, centers]= computeKMeans(X, VTxyz, 100);
    plotData(X, kmeans_training_grp, VTxyz);

    auto error = xt::sum(xt::cast<int>(train_grp) ^ xt::cast<int>(kmeans_training_grp))();
    std::cout << "Number of misclassified patients on training set: " << error << std::endl;

    // TODO: the xt::stddev call is slow... consider replacing?
    std::cout << "TODO: xt::stddev is slow, consider replacing" << std::endl;
    xt::xarray<double> Y = (test_obs - xt::mean(test_obs, 0)) / xt::stddev(test_obs, {0});
    std::cout << "xt::stddev finished" << std::endl;

    auto kmeans_test_grp = testKMeans(Y, VTxyz, centers);
    //std::cout << "kmeans_test_grp= " << kmeans_test_grp << std::endl;
    //std::cout << "test_grp= " << test_grp << std::endl;
    std::cout << "centers=" << centers << std::endl;
    plotData(Y, test_grp, VTxyz);
    plotData(Y, kmeans_test_grp, VTxyz);

    error = xt::sum(xt::cast<int>(test_grp) ^ xt::cast<int>(kmeans_test_grp))();
    std::cout << "Number of misclassified patients on test set: " << error << std::endl;

    return 0;
}
