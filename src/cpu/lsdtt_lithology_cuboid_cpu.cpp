// lsdtt_lithology_cuboid_cpu.cpp
//
// Part of lsdtt3 -- https://github.com/LSDtopotools/lsdtt3
//
// Copyright (C) 2026 Simon M. Mudd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Released under the GNU Affero General Public Licence v3 or later (AGPL-3.0-or-later)
// See LICENSE file or https://www.gnu.org/licenses/agpl-3.0.html

#include "lsdtt_lithology_cuboid_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace lsdtt
{

namespace
{

constexpr double k_pi = 3.141592653589793238462643383279502884;
constexpr double k_grid_tolerance = 1e-9;

bool close_enough(double a, double b)
{
    return std::abs(a - b) <= k_grid_tolerance *
        std::max({1.0, std::abs(a), std::abs(b)});
}

} // namespace

lsdtt_lithology_cuboid_cpu::lsdtt_lithology_cuboid_cpu(
    const lsdtt_raster_cpu& grid,
    std::vector<int> lithology_ids,
    std::vector<float> erodibilities,
    std::vector<float> contact_depths)
    : meta_(grid.get_meta()),
      lithology_ids_(std::move(lithology_ids)),
      erodibilities_(std::move(erodibilities))
{
    if (!meta_.is_valid() || grid.empty())
        throw std::invalid_argument(
            "lsdtt_lithology_cuboid_cpu: reference grid is empty or invalid");

    const std::size_t pixel_count = static_cast<std::size_t>(meta_.n_rows) *
                                    static_cast<std::size_t>(meta_.n_cols);
    if (grid.get_data().size() != pixel_count || grid.get_mask().size() != pixel_count)
        throw std::invalid_argument(
            "lsdtt_lithology_cuboid_cpu: reference grid storage does not match its metadata");

    if (lithology_ids_.empty())
        throw std::invalid_argument(
            "lsdtt_lithology_cuboid_cpu: at least one lithology ID is required");
    if (erodibilities_.empty() ||
        contact_depths.size() + 1 != lithology_ids_.size())
        throw std::invalid_argument(
            "lsdtt_lithology_cuboid_cpu: expected N layer IDs, at least one erodibility, and N-1 contact depths");
    for (int id : lithology_ids_)
        if (id < 0 || static_cast<std::size_t>(id) >= erodibilities_.size())
            throw std::invalid_argument(
                "lsdtt_lithology_cuboid_cpu: each lithology ID must index the erodibility list");
    for (float erodibility : erodibilities_)
        if (!std::isfinite(erodibility) || erodibility <= 0.0f)
            throw std::invalid_argument(
                "lsdtt_lithology_cuboid_cpu: erodibilities must be finite and positive");

    for (std::size_t i = 0; i < contact_depths.size(); ++i)
    {
        if (!std::isfinite(contact_depths[i]) || contact_depths[i] <= 0.0f)
            throw std::invalid_argument(
                "lsdtt_lithology_cuboid_cpu: contact depths must be finite and positive");
        if (i > 0 && !(contact_depths[i] > contact_depths[i - 1]))
            throw std::invalid_argument(
                "lsdtt_lithology_cuboid_cpu: contact depths must be strictly increasing");
    }

    contact_depths_.resize(contact_depths.size());
    for (std::size_t contact = 0; contact < contact_depths.size(); ++contact)
        contact_depths_[contact].assign(pixel_count, contact_depths[contact]);
}

void lsdtt_lithology_cuboid_cpu::require_matching_grid(
    const lsdtt_raster_cpu& raster, const char* raster_name) const
{
    if (raster.empty() || !raster.get_meta().is_valid())
        throw std::invalid_argument(std::string("lsdtt_lithology_cuboid_cpu: ") +
                                    raster_name + " raster is empty or invalid");

    const lsdtt_raster_meta& other = raster.get_meta();
    const bool same_crs = meta_.crs_wkt.empty() || other.crs_wkt.empty() ||
                          meta_.crs_wkt == other.crs_wkt;
    if (meta_.n_rows != other.n_rows || meta_.n_cols != other.n_cols ||
        !close_enough(meta_.cell_size, other.cell_size) ||
        !close_enough(meta_.x_min, other.x_min) ||
        !close_enough(meta_.y_min, other.y_min) || !same_crs)
        throw std::invalid_argument(std::string("lsdtt_lithology_cuboid_cpu: ") +
                                    raster_name + " raster is not on the cuboid grid");

    const std::size_t pixel_count = static_cast<std::size_t>(meta_.n_rows) *
                                    static_cast<std::size_t>(meta_.n_cols);
    if (raster.get_data().size() != pixel_count || raster.get_mask().size() != pixel_count)
        throw std::invalid_argument(std::string("lsdtt_lithology_cuboid_cpu: ") +
                                    raster_name + " raster storage does not match its metadata");
}

std::vector<int> lsdtt_lithology_cuboid_cpu::lithology_at_depth(
    const lsdtt_raster_cpu& depth) const
{
    require_matching_grid(depth, "depth");
    const std::vector<float>& values = depth.get_data();
    const std::vector<std::uint8_t>& mask = depth.get_mask();
    std::vector<int> output(values.size(), 0);

    for (std::size_t pixel = 0; pixel < values.size(); ++pixel)
    {
        if (mask[pixel] == 0 || !std::isfinite(values[pixel]) || values[pixel] < 0.0f)
            continue;

        std::size_t lithology = 0;
        while (lithology < contact_depths_.size() &&
               values[pixel] >= contact_depths_[lithology][pixel])
            ++lithology;
        output[pixel] = lithology_ids_[lithology];
    }
    return output;
}

std::vector<float> lsdtt_lithology_cuboid_cpu::erodibility_at_depth(
    const lsdtt_raster_cpu& depth) const
{
    require_matching_grid(depth, "depth");
    const std::vector<float>& values = depth.get_data();
    const std::vector<std::uint8_t>& mask = depth.get_mask();
    std::vector<float> output(values.size(), static_cast<float>(meta_.no_data_value));

    for (std::size_t pixel = 0; pixel < values.size(); ++pixel)
    {
        if (mask[pixel] == 0 || !std::isfinite(values[pixel]) || values[pixel] < 0.0f)
            continue;

        std::size_t lithology = 0;
        while (lithology < contact_depths_.size() &&
               values[pixel] >= contact_depths_[lithology][pixel])
            ++lithology;
        output[pixel] = erodibilities_[static_cast<std::size_t>(
            lithology_ids_[lithology])];
    }
    return output;
}

void lsdtt_lithology_cuboid_cpu::apply_sine(
    double wavelength, double phase, double amplitude,
    double min_coordinate, double max_coordinate, bool along_y)
{
    if (!std::isfinite(wavelength) || wavelength <= 0.0 ||
        !std::isfinite(phase) || !std::isfinite(amplitude) ||
        std::isnan(min_coordinate) || std::isnan(max_coordinate) ||
        min_coordinate > max_coordinate)
        throw std::invalid_argument(
            "lsdtt_lithology_cuboid_cpu: invalid sine transformation parameters or coordinate range");

    for (std::vector<float>& contact : contact_depths_)
    {
        for (int row = 0; row < meta_.n_rows; ++row)
        {
            for (int col = 0; col < meta_.n_cols; ++col)
            {
                const std::size_t pixel = static_cast<std::size_t>(row) * meta_.n_cols + col;
                // Bounds are edge-relative distances. For y, measure northward
                // from the southern edge; raster rows themselves increase southward.
                const double distance_from_edge = along_y
                    ? (static_cast<double>(meta_.n_rows - row) - 0.5) * meta_.cell_size
                    : (static_cast<double>(col) + 0.5) * meta_.cell_size;
                if (distance_from_edge < min_coordinate ||
                    distance_from_edge > max_coordinate)
                    continue;
                const double coordinate = along_y
                    ? meta_.y_min + (static_cast<double>(row) + 0.5) * meta_.cell_size
                    : meta_.x_min + distance_from_edge;
                const double origin = along_y ? meta_.y_min : meta_.x_min;
                const double displacement = amplitude * std::sin(
                    2.0 * k_pi * (coordinate - origin) / wavelength + phase);
                const double updated = static_cast<double>(contact[pixel]) + displacement;
                if (!std::isfinite(updated) || updated <= 0.0)
                    throw std::invalid_argument(
                        "lsdtt_lithology_cuboid_cpu: transformation produced a non-positive contact depth");
                contact[pixel] = static_cast<float>(updated);
            }
        }
    }
}

void lsdtt_lithology_cuboid_cpu::apply_linear(
    double gradient, double min_coordinate, double max_coordinate, bool along_y)
{
    if (!std::isfinite(gradient) || std::isnan(min_coordinate) ||
        std::isnan(max_coordinate) || min_coordinate > max_coordinate)
        throw std::invalid_argument(
            "lsdtt_lithology_cuboid_cpu: invalid linear transformation parameters or coordinate range");

    for (std::vector<float>& contact : contact_depths_)
    {
        for (int row = 0; row < meta_.n_rows; ++row)
        {
            for (int col = 0; col < meta_.n_cols; ++col)
            {
                const std::size_t pixel = static_cast<std::size_t>(row) * meta_.n_cols + col;
                // Bounds are edge-relative distances. For y, measure northward
                // from the southern edge; raster rows themselves increase southward.
                const double distance_from_edge = along_y
                    ? (static_cast<double>(meta_.n_rows - row) - 0.5) * meta_.cell_size
                    : (static_cast<double>(col) + 0.5) * meta_.cell_size;
                if (distance_from_edge < min_coordinate ||
                    distance_from_edge > max_coordinate)
                    continue;
                const double coordinate = along_y
                    ? meta_.y_min + (static_cast<double>(row) + 0.5) * meta_.cell_size
                    : meta_.x_min + distance_from_edge;
                const double origin = along_y ? meta_.y_min : meta_.x_min;
                const double updated = static_cast<double>(contact[pixel]) +
                                       gradient * (coordinate - origin);
                if (!std::isfinite(updated) || updated <= 0.0)
                    throw std::invalid_argument(
                        "lsdtt_lithology_cuboid_cpu: transformation produced a non-positive contact depth");
                contact[pixel] = static_cast<float>(updated);
            }
        }
    }
}

void lsdtt_lithology_cuboid_cpu::apply_sine_y(
    double wavelength, double phase, double amplitude,
    double min_coordinate, double max_coordinate)
{
    apply_sine(wavelength, phase, amplitude, min_coordinate, max_coordinate, true);
}

void lsdtt_lithology_cuboid_cpu::apply_sine_x(
    double wavelength, double phase, double amplitude,
    double min_coordinate, double max_coordinate)
{
    apply_sine(wavelength, phase, amplitude, min_coordinate, max_coordinate, false);
}

void lsdtt_lithology_cuboid_cpu::apply_linear_y(
    double gradient, double min_coordinate, double max_coordinate)
{
    apply_linear(gradient, min_coordinate, max_coordinate, true);
}

void lsdtt_lithology_cuboid_cpu::apply_linear_x(
    double gradient, double min_coordinate, double max_coordinate)
{
    apply_linear(gradient, min_coordinate, max_coordinate, false);
}

} // namespace lsdtt
