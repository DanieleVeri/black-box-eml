import sys
import numpy as np
import docplex.mp.model as cpx
from . import base

class CplexBackend(base.Backend):
    """ Backend for CPLEX solver

    Attributes
    ---------
        _ml_tol  : float
            Tollerance

    Parameters
    ----------
        ml_tol :float)
            Tollerance

    """
    def __init__(self, ml_tol=1e-4):
        self._ml_tol = ml_tol
        self.vars = {}
        super(CplexBackend, self).__init__()

    def add_cst(self, mdl, cst, name=None):
        mdl.add_constraint(cst, name)

    def get_lb(self, var):
        return var.lb

    def get_ub(self, var):
        return var.ub

    def const_eps(self, mdl):
        """ Get tollerance

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model

        Returns
        -------
            Tollerance : float
                Tollerance

        """
        return self._ml_tol

    def var_cont(self, mdl, lb, ub, name=None):
        """ Creates continuous variable in the model

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            lb : float)
                Lower bound of the variable
            ub :float
                Upper bound of the variable
            name : string
                Name of the variable (default None)

        Returns
        -------
            Continuos Variable : :obj:`docplex.mp.linear.Var``
                Continuos variable with specified bounds and name

        """
        # Convert bounds in a cplex friendly format
        lb = lb if lb != -float('inf') else -mdl.infinity
        ub = ub if ub != float('inf') else mdl.infinity
        v = mdl.continuous_var(lb=lb, ub=ub, name=name)
        self.vars[name] = v
        return v

    def var_int(self, mdl, lb, ub, name=None):
        """ Creates integer variable in the model

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            lb : float)
                Lower bound of the variable
            ub :float
                Upper bound of the variable
            name : string
                Name of the variable (default None)

        Returns
        -------
            Continuos Variable : :obj:`docplex.mp.linear.Var``
                Continuos variable with specified bounds and name

        """
        # Convert bounds in a cplex friendly format
        lb = lb if lb != -float('inf') else -mdl.infinity
        ub = ub if ub != float('inf') else mdl.infinity
        v = mdl.integer_var(lb=lb, ub=ub, name=name)
        self.vars[name] = v
        return v

    def var_bin(self, mdl, name=None):
        """ Creates continuous variable in the model

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            name : string)
                Name of the variable (default None)

        Returns
        -------
            Binary Variable : :obj:`docplex.mp.linear.Var``
                Binary Variable

        """
        v = mdl.binary_var(name=name)
        self.vars[name] = v
        return v

    def xpr_scalprod(self, mdl, coefs, terms):
        """ Scalar product of varibles and coefficients

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            coefs : list(float)
                List of coefficients
            terms : list(:obj:docplex.mp.linear.Var]):
                List of variables

        Returns
        -------
            Linear Expression : :obj:`docplex.mp.LinearExpr()`
                Linear expression representing the linear combination
                of terms and coefficients or 0

        """
        return sum(c * x for c, x in zip(coefs, terms))

    def xpr_sum(self, mld, terms):
        """ Sum of variables

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            terms : list(:obj:docplex.mp.linear.Var)
                List of variables

        Returns
        -------
            Linear Expression : :obj:`docplex.mp.LinearExpr()`
                Linear expression representing the sum of all
                the term in input

        """
        return sum(terms)

    def xpr_eq(self, mdl, left, right):
        """ Creates an equality constraint between two variables

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            left : :obj:docplex.mp.linear.Var
                Variable
            right : :obj:docplex.mp.linear.Var
                Variable

        Returns
        -------
            Equality constraint : :obj:`docplex.mp.constr.LinearConstraint`
                Equality contraint between the two variables in input

        """
        return left == right

    def xpr_leq(self, mdl, left, right):
        """ Creates an inequality constraint between two variables

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            left : :obj:docplex.mp.linear.Var
                Variable
            right : :obj:docplex.mp.linear.Var
                Variable

        Returns
        -------
            Inequality constraint : :obj:`docplex.mp.constr.LinearConstraint`
                Inequality contraint between the two variables in input

        """
        return left <= right

    def cst_eq(self, mdl, left, right, name=None):
        """ Add to the model equality constraint between two variables

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            left : :obj:docplex.mp.linear.Var
                Variable
            right : :obj:docplex.mp.linear.Var
                Variable
            name : string
                Name of the constraint

        Returns
        -------
            Equality constraint : :obj:`docplex.mp.constr.LinearConstraint`
                Equality contraint between the two variables in input


        """
        return mdl.add_constraint(left == right, ctname=name)

    def cst_leq(self, mdl, left, right, name=None):
        """ Add to the model a lowe or equal constraint between two variables

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            left : :obj:docplex.mp.linear.Var
                Variable
            right : :obj:docplex.mp.linear.Var
                Variable
            name : string
                Name of the constraint

        Returns
        -------
            Lower or equal constraint : :obj:`docplex.mp.constr.LinearConstraint`
                Lowe or equal contraint between the two variables in input


        """
        return mdl.add_constraint(left <= right, ctname=name)

    def cst_indicator(self, mdl, trigger, val, cst, name=None):
        """ Add an indicator to the model

        An indicator constraint links (one-way) the value of a
        binary variable to the satisfaction of a linear constraint

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            trigger : :obj:`docplex.mp.Var`
                Binary Variable
            val : int
                Active value, used to trigger the satisfaction
                of the constraint
            cst : :obj:`docplex.mp.constr.LinearConstraint`
                Linear constraint
            name : string
                Name of the constraint

        Returns
        -------
            Indicator constraint : :obj:`docplex.mp.constr.IndicatorConstraint`
                Indicator constraint between the trigger and the linear
                constraint in input

        """
        return mdl.add_indicator(trigger, cst, val, name=name)

    def get_obj(self, mdl):
        """ Returns objextive expression

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model

        Returns
        -------
            Objective and expression : (string, )
                'min' if the objective function is to be minimized,
                'max otherwise.
                The expression repesenting the objective function

        """
        sense = 'min' if mdl.is_minimized() else 'max'
        xpr = mdl.get_objective_expr()
        return sense, xpr

    def set_obj(self, mdl, sense, xpr):
        """ Sets the objective function

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model
            sense : string
                Represents the objective, 'min' or 'max'
            xpr :
                Expression representing the objective function

        Returns
        -------
            None

        """
        mdl.set_objective(sense, xpr)

    def solve(self, mdl, timelimit):
        """ Solves the problem

        Parameters
        -----------
            mdl : :obj:`docplex.mp.model.Model`
                            Cplex model
            timelimit : int
                time limit in seconds for the solver

        Returns
        -------
            Solution : :obj:`docplex.mp.solution.SolveSolution`
                A solution if the problem is feasible, the status of the
                of the solver otherwise

        """
        mdl.set_time_limit(max(1, timelimit))
        res = mdl.solve()
        stime = mdl.solve_details.time
        status = 'infeasible' if res is None else res.solve_details.status
        obj = None if res is None else res.objective_value
        bound = mdl.solve_details.best_bound
        var_dict = {k: res[k] for k, v in self.vars.items()}
        return {
            'status':status,
            'obj': obj,
            'time': stime,
            'bound': bound,
            'vars': var_dict
        }

    def new_model(self, mdl=None, name=None):
        """ Creates a new model

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model (default None)
            name : string
                Name of the model (default None)

        Returns
        -------
            Model : :obj:`docplex.mp.model.Model`
                Cplex model

        """
        return cpx.Model()

    def set_determinism(self, mdl, seed=42):
        # mdl.parameters.dettimelimit = self.solver_timeout * 1e3
        # mdl.parameters.tune.dettimelimit = self.solver_timeout * 1e3
        # mdl.parameters.threads = 1
        mdl.parameters.parallel = 1
        mdl.parameters.randomseed = seed

    def set_extensive_log(self, mdl):
        mdl.context.solver.verbose = 5
        mdl.context.solver.log_output = True
        mdl.print_information()
        # Write the whole model on a txt file
        box = sys.stdout
        sys.stdout = open('cplex_model.txt', 'w')
        mdl.prettyprint()
        sys.stdout.close()
        sys.stdout = box
