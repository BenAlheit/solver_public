#ifndef SOLVER_PROBLEM_H
#define SOLVER_PROBLEM_H

#include "Stage.h"
#include "../materials/Material.h"
#include "FEMesh.h"

template<unsigned int dim>
class Problem{
public:
    Problem(const string & name,
            FEMesh<dim> * mesh,
            const vector<Stage<dim>> & stages,
            Time* time)
            : name(name)
            , mesh(mesh)
            , stages(stages)
            , time(time)
    {
        system(("mkdir -p " + name).c_str());
        system(("mkdir -p " + name + "/VolumeAverages").c_str());
    };

    bool step(AffineConstraints<double> & constraints);

    void next_stage(){
        current_stage++;
        stages.at(current_stage).update_time_values();
    };

    bool more_stages(){return current_stage < stages.size();}
    bool first_stage(){return current_stage == 0;}
    bool last_stage(){return current_stage == stages.size()-1;}

    FEMesh<dim>* get_mesh() const {return mesh;};
    const Stage<dim> & get_stage() const {return stages.at(current_stage);};
    Time* get_time() {return time;};
    string get_name() const {return name;};

protected:

private:
    const string name;
    FEMesh<dim> * mesh;
    vector<Stage<dim>> stages;
    unsigned int current_stage = 0;
    Time* time;

    void set_increment_constraints(AffineConstraints<double> & constraints,
                                   double inc_pct);
//    void fix_increment_constraints(AffineConstraints<double> & constraints);

};

template<unsigned int dim>
bool Problem<dim>::step(AffineConstraints<double> & constraints) {
    bool output = time->increment();
    set_increment_constraints(constraints, time->delta_stage_pct());
    return output;
}

template<unsigned int dim>
void Problem<dim>::set_increment_constraints(AffineConstraints<double> &constraints,
                                             double inc_pct) {
    constraints.clear();
    for(const auto & dbc : stages.at(current_stage).get_dbcs()){
        dbc->apply_bc(mesh->get_dof_handler(), constraints, inc_pct);
    }
    DoFTools::make_hanging_node_constraints(mesh->get_dof_handler(), constraints);
    constraints.close();
}

#endif //SOLVER_PROBLEM_H
