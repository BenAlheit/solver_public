#include <iostream>
#include "Solver.h"
#include "SolverExample.h"


#include "examples/CubeWithIMP.h"


#include "examples/Cube.h"
#include "examples/ElastoPlasticCube.h"
#include "examples/QuarterCylinder.h"
#include "examples/NeckingCylinder.h"
#include "examples/PSC.h"

#include "examples/check-approximate-tangent/CheckApproximateTangentElastic.h"
#include "examples/check-approximate-tangent/CheckApproximateTangentViscoelastic.h"

#include "examples/cyclical-tension-compression/ElasticCyclicalTensionCompression.h"
#include "examples/cyclical-tension-compression/ViscoCyclicalTensionCompression.h"
#include "examples/cyclical-tension-compression/ExplicitRateDependentPlasticityCyclicalTensionCompression.h"

#include "examples/cube-with-sphere/CubeWithSphere.h"
#include "examples/cube-with-sphere/CubeWithSpherePBC.h"
#include "examples/cube-with-sphere/ViscoplasticCubeWithSpherePBC.h"

#include "examples/polycrystal/PolycrystalPBC.h"
#include "examples/polycrystal-imp/PolycrystalIMPPBC.h"

// TODO Documentation
// TODO Documentation
// TODO Documentation
// TODO Documentation
// TODO Documentation

// TODO Viscocrystalplastic material
// TODO Fix parallelization

int main(int argc, char **argv){
//    MultithreadInfo::set_thread_limit(1);

    using namespace dealii;
    using namespace SolverExample;
    const unsigned int dim = 3;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

//    PSC<dim> problem = PSC<dim>();


//    CubeWithIMP<dim> problem = CubeWithIMP<dim>();

//    PolycrystalPBC<dim> problem = PolycrystalPBC<dim>();
//    PolycrystalIMPPBC<dim> problem = PolycrystalIMPPBC<dim>();


//    CheckApproximateTangentElastic<dim> problem = CheckApproximateTangentElastic<dim>();
//    CheckApproximateTangentViscoelastic<dim> problem = CheckApproximateTangentViscoelastic<dim>();

//    ElastoPlasticCube<dim> problem = ElastoPlasticCube<dim>();
//    QuarterCylinder<dim> problem = QuarterCylinder<dim>();
//    NeckingCylinder<dim> problem = NeckingCylinder<dim>();


//    ElasticCyclicalTensionCompression<dim> problem = ElasticCyclicalTensionCompression<dim>();
//    ViscoCyclicalTensionCompression<dim> problem = ViscoCyclicalTensionCompression<dim>();
//    ExplicitRateDependentPlasticityCyclicalTensionCompression<dim> problem
//    = ExplicitRateDependentPlasticityCyclicalTensionCompression<dim>();

//    Cube<dim> problem = Cube<dim>();
//    CubeWithSphere<dim> problem = CubeWithSphere<dim>();
//    CubeWithSpherePBC<dim> problem = CubeWithSpherePBC<dim>();
    ViscoplasticCubeWithSpherePBC<dim> problem = ViscoplasticCubeWithSpherePBC<dim>();


//    ElasticProblem<2> elastic_problem;
//    elastic_problem.run();

    return 0;
}
