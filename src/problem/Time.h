#ifndef SOLVER_TIME_H
#define SOLVER_TIME_H

#include <deal.II/base/function_time.h>
#include <exception>
#include "../utils/utils.h"
#include "Stage.h"

class IncrementExcededException : public exception {
    const char *what() const throw() override {
        return "The maximum number of time increments for this stage has been exceded.";
    }
} increment_exceded;

class PredictIncrementExcededException : public exception {
public:
    PredictIncrementExcededException(unsigned int max_incs, unsigned int predicted_incs) :
            max_incs(max_incs), predicted_incs(predicted_incs) {
        message = "The number of increments to complete the stage with the current step size is " +
                  to_string(predicted_incs) + ". However, maximum number of increments allowed is set to" +
                  to_string(max_incs) + ".";
    };

    const char *what() const throw() override {
        return message.c_str();
    }

private:
    unsigned int max_incs;
    unsigned int predicted_incs;
    string message;
};


class Time {
public:
    Time() {}

    Time(const double time_end,
         unsigned int n_steps,
         unsigned int n_steps_out = 0,
         const double time_start = 0)
            : timestep(0),
              time_end(time_end),
              delta_t((time_end - time_start) / n_steps),
              output_delta_t((n_steps_out == 0) ?
                             (time_end - time_start) / n_steps
                                                : (time_end - time_start) / n_steps_out),
              next_output(time_start + output_delta_t), n_steps(n_steps), n_steps_out(n_steps_out) {}

    Time(const double time_end, const double delta_t)
            : timestep(0), time_end(time_end), delta_t(delta_t), time(0.0) {}

    virtual ~Time() = default;

    double current() const {
        return time;
    }

    double end() const {
        return time_end;
    }

    double get_original_delta_t() const {
        return delta_t;
    }

    double get_current_delta_t() const {
        return current_delta_t;
    }

    void set_delta_t(const double &new_delta_t) {
        delta_t = new_delta_t;
    }

    unsigned int get_n_steps() const {
        return n_steps;
    }

    unsigned int get_n_steps_out() const {
        return n_steps_out;
    }

    unsigned int get_timestep() const {
        return timestep;
    }

    unsigned int get_stage() const {
        return stage;
    }

    bool increment() {
        if (cut) {
            if (update_with_cut) delta_t = delta_t / pow(2, n_cuts);
            n_cuts = 0;
            cut = false;
        }
        bool output = false;
        double next_time = this->time + delta_t;
        current_delta_t = delta_t;
        if (next_time > time_end) {
            current_delta_t = time_end - this->time;
            this->time = time_end;
            output = true;
        } else if (next_time >= next_output || almost_equals(next_time, next_output)) {
            output = true;
            current_delta_t = next_output - this->time;
            this->time = next_output;
            next_output += output_delta_t;
            output_step++;
        } else this->time = next_time;

        timestep++;

        if (set_max_incs && timestep > max_incs) throw increment_exceded;
        if (predictive_max_incs) {
            unsigned int tot_incs = timestep + (int) (time_end - time) / delta_t;
            if (tot_incs > max_incs) throw PredictIncrementExcededException(max_incs, tot_incs);
        }

        return output;
    }

    void next_stage(const double end_time, const double dt) {
        stage++;
        this->time_end = end_time;
        this->delta_t = dt;
    }

    void set_time_end(const double &new_time_end) {
        this->time_end = new_time_end;
    }

    void cut_step() {
        cut = true;
        n_cuts++;
        time -= delta_t / pow(2, n_cuts);
    }

    void increase_dt() {
        delta_t = delta_t * 1.5;
    }

    double stage_pct() {
        return (time - time_start) / (time_end - time_start);
    }

    double delta_stage_pct() {
        return current_delta_t / (time_end - time_start);
    }

    void next_stage(const double &end_time,
                    const unsigned int &n_steps,
                    const unsigned int &n_out){
        this->time_end = end_time;
        this->n_steps = n_steps;
        this->n_steps_out = n_out;
        this->time_start = this->current();
        this->timestep = 0;
        this->stage++;

        this->delta_t = (end_time - this->current()) / n_steps;
        this->output_delta_t = (end_time - this->current()) / n_out;
        this->next_output = this->current() + this->output_delta_t;
        this->current_delta_t = this->delta_t;
    }

private:
    unsigned int n_steps, n_steps_out;
    unsigned int timestep = 0;
    unsigned int stage = 0;
    unsigned int n_cuts = 0;
    bool cut = false;
    bool update_with_cut = true;
    double time_start = 0;
    double time_end = 0;
    double delta_t_original = 0;
    double delta_t = 0;
    double current_delta_t = 0;
    unsigned int output_step = 0;
    double output_delta_t = 0;
    double next_output = 0;
    double time = 0;
    unsigned int max_incs = 0;
    bool set_max_incs = false;
    bool predictive_max_incs = false;
};


#endif //SOLVER_TIME_H
