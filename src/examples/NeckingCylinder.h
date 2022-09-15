//
// Created by alhei on 2022/09/01.
//

#ifndef SOLVER_NECKINGCYLINDER_H
#define SOLVER_NECKINGCYLINDER_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include "../utils/utils.h"
#include "../materials/OutputFlags.h"
#include "../problem/FEMesh.h"
#include "../Solver.h"

using namespace std;
using namespace dealii;


template<unsigned int dim>
class NeckingCylinder {
public:
    explicit NeckingCylinder(unsigned int n_refinements = 1);

    Triangulation<dim> triangulation;
    map<unsigned int, Material<dim> *> materials;
    vector<Stage<dim>> stages;
    Time time;

private:

    void make_mesh();

    const unsigned int n_refinements;

    Point<dim> p1 = Point<dim>({0, 0, 0});
    Point<dim> p2 = Point<dim>({1, 1, 1});
    double r = 6.413;
    double l = 53.334/2.;

    array<unsigned int, dim> range;
    const unsigned int order = 1;
    const double geom_error_pct = 1.8;

};

template<unsigned int dim>
NeckingCylinder<dim>::NeckingCylinder(unsigned int n_refinements) : n_refinements(n_refinements) {
    iota(range.begin(), range.end(), 0);

    make_mesh();
    string name = "stricter-necking-cylinder";
//    const double kappa = 44.12e3;
//    const double mu = 16.92e3;
//    materials[/*material id*/ 0] = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);

    constexpr double E = 206.9e3;
    constexpr double v = 0.29;

    constexpr double kappa = E / (3 * (1-2*v));
    constexpr double mu = E / (2 * (1 + v));

    auto elastic_component = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);

    auto flow_rule = new MaximumDissipation<dim>();
    auto hardening_law = new VoceWLinear<dim>(0.45e3, 0.715e3, 16.93, 0.12924e3);
    auto yield_surface = new VonMises<dim>(hardening_law);
    auto plasticity_theory = new PlasticityTheory<dim>(yield_surface, flow_rule);

    auto material = new RateIndependentPlasticity<dim>(elastic_component, plasticity_theory);

    materials[/*material id*/ material->get_id()] = material;

    unsigned int n_steps = 200;
    unsigned int n_out = 200;
    double pull = 7;

    time = Time(/*end_time*/ 1,
                             n_steps,
                             n_out);

    vector<DirichletBoundaryCondition<dim> *> dbcs;
    for (const auto &i: range)
        dbcs.push_back(new Slider<dim>(1 + i, i));

    dbcs.push_back(new DirichletBoundaryCondition<dim>( /*boundary_id*/ 4,
            /*components*/ vector<unsigned int>({0}),
            /*magnitudes*/ vector<double>({pull})));

    vector<unsigned int> material_ids({0});

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> scalar_outputs;
    scalar_outputs[OutputFlags::ScalarOutputFlag::J] = material_ids;
    scalar_outputs[OutputFlags::ScalarOutputFlag::P] = material_ids;
    scalar_outputs[OutputFlags::ScalarOutputFlag::EP] = material_ids;
    scalar_outputs[OutputFlags::ScalarOutputFlag::DLAM] = material_ids;
    scalar_outputs[OutputFlags::ScalarOutputFlag::FPLAS] = material_ids;

    map<OutputFlags::VectorOutputFlag, vector<unsigned int>> vector_outputs;
    vector_outputs[OutputFlags::VectorOutputFlag::NB1] = material_ids;
    vector_outputs[OutputFlags::VectorOutputFlag::NB2] = material_ids;
    vector_outputs[OutputFlags::VectorOutputFlag::NB3] = material_ids;

    map<OutputFlags::TensorOutputFlag, vector<unsigned int>> tensor_outputs;
    tensor_outputs[OutputFlags::TensorOutputFlag::F] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::FP] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::FE] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::STRAIN] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::STRESS] = material_ids;

    vector<OutputFlags::MeshOutputFlag> mesh_outputs({OutputFlags::MeshOutputFlag::MATERIAL_ID,
                                                      OutputFlags::MeshOutputFlag::BOUNDARY_ID});

    Stage<dim> first_stage(&time,
                           dbcs,
                           vector<NeumannBoundaryCondition<dim> *>({}),
                           scalar_outputs,
                           vector_outputs,
                           tensor_outputs,
                           mesh_outputs);

    stages.push_back(first_stage);
    FEMesh<dim> mesh(order,
                     materials,
                     triangulation);

    Problem<dim> problem(name, &mesh, stages, &time);
    Solver::Solver<dim> solver(&problem);
    solver.solve();

}

template<unsigned int dim>
void NeckingCylinder<dim>::make_mesh() {
    Triangulation<dim> inti_triangulation;
    GridGenerator::subdivided_cylinder(inti_triangulation, (unsigned int) 2 * (2 + l / r), r, l);
    set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;
    inti_triangulation.refine_global();


    for (const auto &cell: inti_triangulation.active_cell_iterators())
        if (cell->center()[0] < 0 || cell->center()[1] < 0 || cell->center()[2] < 0)
            cells_to_remove.insert(cell);

    GridGenerator::create_triangulation_with_removed_cells(inti_triangulation, cells_to_remove, triangulation);

    for (const auto &cell: triangulation.cell_iterators())
        cell->set_material_id(0);

    triangulation.reset_all_manifolds();
    triangulation.set_all_manifold_ids(0);

    Point<dim> vert_yz;

    for (auto &face: triangulation.active_face_iterators()) {
        if (face->at_boundary()) {
            for (const auto &i: range) {
                if (almost_equals(face->center()[i], p1[i])) {
                    face->set_boundary_id(1 + i);
                    break;
                }
            }
            if (almost_equals(face->center()[0], l)) {
                face->set_boundary_id(1 + dim);
            }
            bool face_at_sphere_boundary = true;
            for (const auto v: face->vertex_indices()) {
                vert_yz = face->vertex(v);
                vert_yz[0] = 0;
                if (not almost_equals(vert_yz.norm(), r)) {
                    face_at_sphere_boundary = false;
                    break;
                }
            }
            if (face_at_sphere_boundary)
                face->set_all_manifold_ids(1);
        }
    }

    triangulation.set_manifold(1, CylindricalManifold<dim>());

    if (n_refinements > 0) triangulation.refine_global(n_refinements);

    double l_fith = l / 5.;
    double l_half = l / 2.;
    double l_rest = l - l_fith;


    vector<typename Triangulation<dim>::vertex_iterator> moved_vertices;

    for (const auto &cell: triangulation.active_cell_iterators()) {
        for (const auto i: cell->vertex_indices()) {
            if (find(moved_vertices.begin(), moved_vertices.end(), cell->vertex_iterator(i)) == moved_vertices.end()) {
                Point<dim> &v = cell->vertex(i);
                if (almost_equals(v(0), 0)){
                    v *= (1-geom_error_pct/100.);
                }

                if (v(0) < l_half) {
                    v(0) = (v(0) / l_half) * l_fith;
                } else {
                    v(0) = l_fith + l_rest * (v(0) - l_half) / l_half;
                }
                moved_vertices.push_back(cell->vertex_iterator(i));
            }
        }
    }

//    for (const auto &cell: triangulation.active_cell_iterators()) {
//        for (const auto i: cell->vertex_indices()) {
//            if (find(moved_vertices.begin(), moved_vertices.end(), cell->vertex_iterator(i)) == moved_vertices.end()) {
//                Point<dim> &v = cell->vertex(i);
//                if (almost_equals(v(0), 0) && almost_equals(v.norm(), r)){
//                    v *= (1-geom_error_pct/100.);
//                }
//                moved_vertices.push_back(cell->vertex_iterator(i));
//            }
//        }
//    }

}

#endif //SOLVER_NECKINGCYLINDER_H
