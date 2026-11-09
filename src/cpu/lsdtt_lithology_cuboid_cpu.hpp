// lsdtt_lithology_cuboid_cpu.hpp
//
// Part of lsdtt3 -- https://github.com/LSDtopotools/lsdtt3
//
// Copyright (C) 2026 Simon M. Mudd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Released under the GNU Affero General Public Licence v3 or later (AGPL-3.0-or-later)
// See LICENSE file or https://www.gnu.org/licenses/agpl-3.0.html
//
// -----------------------------------------------------------------------------
// A raster-grid lithology cuboid with spatially varying contact depths.
// -----------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <vector>

#include "lsdtt_raster_cpu.hpp"

namespace lsdtt
{

class lsdtt_lithology_cuboid_cpu
{
public:
    // Contacts are supplied top-to-bottom (strictly increasing positive depth).
    // A contact separates lithology_ids[i] from lithology_ids[i + 1]. IDs are
    // indices into erodibilities and may repeat. The supplied depths are
    // initially horizontal and are expanded over the grid.
    lsdtt_lithology_cuboid_cpu(
        const lsdtt_raster_cpu& grid,
        std::vector<int> lithology_ids,
        std::vector<float> erodibilities,
        std::vector<float> contact_depths);

    [[nodiscard]] const lsdtt_raster_meta& get_meta() const noexcept
    {
        return meta_;
    }

    [[nodiscard]] const std::vector<int>& lithology_ids() const noexcept
    {
        return lithology_ids_;
    }

    [[nodiscard]] const std::vector<float>& erodibilities() const noexcept
    {
        return erodibilities_;
    }

    [[nodiscard]] std::size_t contact_count() const noexcept
    {
        return contact_depths_.size();
    }

    // Each contact surface is stored in flat raster order, one depth per cell.
    [[nodiscard]] const std::vector<std::vector<float>>& contact_depths() const noexcept
    {
        return contact_depths_;
    }

    // A depth exactly on a contact belongs to the lithology below it.
    [[nodiscard]] std::vector<int> lithology_at_depth(
        const lsdtt_raster_cpu& depth) const;

    // Per-cell K values in flat raster order, ready for stream-power calls.
    [[nodiscard]] std::vector<float> erodibility_at_depth(
        const lsdtt_raster_cpu& depth) const;

    // Apply a sinusoidal vertical displacement to every contact. Bounds are
    // inclusive distances northward from the southern (y) or eastward from the
    // western (x) grid edge, in grid
    // units, independent of the georeferenced origin. Wavelength is positive;
    // amplitude uses depth units and phase is in radians.
    void apply_sine_y(double wavelength, double phase, double amplitude,
                      double min_coordinate, double max_coordinate);
    void apply_sine_x(double wavelength, double phase, double amplitude,
                      double min_coordinate, double max_coordinate);

    // Apply a linear vertical displacement to every contact. Bounds are
    // inclusive distances northward from the southern (y) or eastward from the
    // western (x) grid edge, in grid
    // units. The gradient is change in depth per grid unit along that axis.
    void apply_linear_y(double gradient, double min_coordinate,
                        double max_coordinate);
    void apply_linear_x(double gradient, double min_coordinate,
                        double max_coordinate);

private:
    void require_matching_grid(const lsdtt_raster_cpu& raster,
                               const char* raster_name) const;
    void apply_sine(double wavelength, double phase, double amplitude,
                    double min_coordinate, double max_coordinate, bool along_y);
    void apply_linear(double gradient, double min_coordinate,
                      double max_coordinate, bool along_y);

    lsdtt_raster_meta meta_;
    std::vector<int> lithology_ids_;
    std::vector<float> erodibilities_;
    // Contact-major layout: contact_depths_[contact][row * n_cols + col].
    std::vector<std::vector<float>> contact_depths_;
};

} // namespace lsdtt
