import time
from . import base
from ortools.linear_solver import pywraplp

class OrtoolsBackend(base.Backend):
    """ Backend for ortools solver

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
        self.obj_expr = None
        self.vars = {}
        super(OrtoolsBackend, self).__init__()

    def get_lb(self, var):
        return var.Lb()

    def get_ub(self, var):
        return var.Ub()

    def add_cst(self, mdl, cst, name=''):
        mdl.Add(cst, name)

    def const_eps(self, mdl):
        """ Get tollerance

        Parameters
        ----------
            mdl : ortools solver
                ortools model

        Returns
        -------
            Tollerance : float
                Tollerance

        """
        return self._ml_tol

    def var_cont(self, mdl, lb, ub, name=''):
        """ Creates continuous variable in the model

        Parameters
        ----------
            mdl : ortools solver
                ortools model
            lb : float)
                Lower bound of the variable
            ub :float
                Upper bound of the variable
            name : string
                Name of the variable (default None)

        Returns
        -------
            Continuos Variable : ortools continuous variable
                Continuos variable with specified bounds and name

        """
        # Convert bounds in a ortools friendly format
        lb = lb if lb != float('-inf') else -mdl.infinity()
        ub = ub if ub != float('+inf') else mdl.infinity()
        # Build the variable
        v = mdl.NumVar(lb=lb, ub=ub, name=name)
        self.vars[name] = v
        return v

    def var_int(self, mdl, lb, ub, name=''):
        """ Creates integer variable in the model

        Parameters
        ----------
            mdl : ortools solver
                ortools model
            lb : float)
                Lower bound of the variable
            ub :float
                Upper bound of the variable
            name : string
                Name of the variable (default None)

        Returns
        -------
            Continuos Variable : ortools continuous variable
                Continuos variable with specified bounds and name

        """
        # Convert bounds in a ortools friendly format
        lb = lb if lb != float('-inf') else -mdl.infinity()
        ub = ub if ub != float('+inf') else mdl.infinity()
        # Build the variable
        v = mdl.IntVar(lb=lb, ub=ub, name=name)
        self.vars[name] = v
        return v

    def var_bin(self, mdl, name=''):
        """ Creates binary variable in the model

        Parameters
        ----------
            mdl : ortools solver
                ortools model
            name : string)
                Name of the variable (default None)

        Returns
        -------
            Binary Variable : ortools binary varible
                Binary Variable

        """
        v = mdl.IntVar(0, 1, name=name)
        self.vars[name] = v
        return v

    def xpr_scalprod(self, mdl, coefs, terms):
        """ Scalar product of varibles and coefficients

        Parameters
        ----------
            mdl : ortools solver
                ortools modek
            coefs : list(float)
                List of coefficients
            terms : list(ortools var):
                List of variables

        Returns
        -------
            Linear Expression : ortools linear expression
                Linear expression representing the linear combination
                of terms and coefficients or 0

        """
        return sum(c * x for c, x in zip(coefs, terms))

    def xpr_sum(self, mld, terms):
        """ Sum of variables

        Parameters
        ----------
            mdl : ortools solver
                ortools model
            terms : list(ortools variables)
                List of variables

        Returns
        -------
            Linear Expression : ortoools linear expression
                Linear expression representing the sum of all
                the term in input

        """
        return sum(terms)

    def xpr_eq(self, mdl, left, right):
        """ Creates an equality constraint between two variables

        Parameters
        ----------
            mdl : ortools solver
                ortools model
            left : ortools varaible
                Variable
            right : ortools varaible
                Variable

        Returns
        -------
            Equality constraint : ortools linear constraint
                Equality contraint between the two variables in input

        """
        return left == right

    def xpr_leq(self, mdl, left, right):
        """ Creates an inequality constraint between two variables

        Parameters
        ----------
            mdl : ortools solver
                ortools model
            left : ortools varaible
                Variable
            right : ortools varaible
                Variable

        Returns
        -------
            Inequality constraint : ortools linear constraint
                Inequality contraint between the two variables in input

        """
        return left <= right

    def cst_eq(self, mdl, left, right, name=''):
        """ Add to the model equality constraint between two variables

        Parameters
        ----------
            mdl : ortools solver
                ortools model
            left : ortools variable
                Variable
            right : ortools variable
                Variable
            name : string
                Name of the constraint

        Returns
        -------
            Equality constraint : ortools linear constraint
                Equality contraint between the two variables in input


        """
        return mdl.Add(left == right, name=name)

    def cst_leq(self, mdl, left, right, name=''):
        """ Add to the model a lowe or equal constraint between two variables

        Parameters
        ----------
            mdl : ortools solver
                ortools model
            left : ortools variable
                Variable
            right : ortools variable
                Variable
            name : string
                Name of the constraint

        Returns
        -------
            Lower or equal constraint : ortools linear constraint
                Lowe or equal contraint between the two variables in input


        """
        return mdl.Add(left <= right, name=name)

    def cst_indicator(self, mdl, trigger, val, cst, name=''):
        """ Add an indicator to the model

        An indicator constraint links (one-way) the value of a
        binary variable to the satisfaction of a linear constraint

        Parameters
        ----------
            mdl : ortools model
                ortools model
            trigger : threshold
                numerical value (float or int)
            val : int
                Active value, used to trigger the satisfaction
                of the constraint
            cst : ortools linear constraint
                Linear constraint
            name : string
                Name of the constraint

        Returns
        -------
            Indicator constraint : ortools indicator variable
                Indicator constraint between the trigger and the linear
                constraint in input

        """
        if val not in (0,1):
            raise ValueError("val must be 0 or 1 in an indicator constraint")

        # Extract information from the constraint
        clb = cst._LinearConstraint__lb
        cub = cst._LinearConstraint__ub
        xpr = cst._LinearConstraint__expr
        # Obtain bounds for the variables in the constraint
        coeffs, xvars, xlb, xub = [], [], [], []
        Ml, Mu = 0, 0
        for x, c in xpr.GetCoeffs().items():
            coeffs.append(c) # store the coefficient
            xvars.append(x) # store the variable
            xlb.append(x.lb()) # store the variable lb
            xub.append(x.ub()) # store the variable ub
            # Sanity check
            invalid = (mdl.infinity(),mdl.infinity())
            if x.lb() in invalid or x.ub() in invalid:
                es = f'Infinite bounds detected for variable {str(x)}. '
                es += 'The or-tools backend cannot handle ReLU activation '
                es += 'functions unless bounds are pre-computed. You can '
                es += 'use the net.process.ibr_bounds method to propagate '
                es += 'network input bounds and get rid of this error.'
                raise ValueError(es)
            # Add a term to the global bounds
            if c >= 0:
                Ml += c * x.lb()
                Mu += c * x.ub()
            else:
                Ml += c * x.ub()
                Mu += c * x.lb()

        # Linearize the indicator constraint
        if val == 0:
            if cub != float('inf'):
                mdl.Add(xpr <= cub + trigger * (Mu - cub))
            if clb != -float('inf'):
                mdl.Add(xpr >= clb + trigger * (Ml - clb))
        else:
            if cub != float('inf'):
                mdl.Add(xpr <= cub + (1-trigger) * (Mu - cub))
            if clb != -float('inf'):
                mdl.Add(xpr >= clb + (1-trigger) * (Ml - clb))

    def set_obj(self, mdl, sense, xpr):
        """ Sets the objective function

        Parameters
        ----------
            mdl : ortools solver
                ortools model
            sense : string
                Represents the objective, 'min' or 'max'
            xpr :
                Expression representing the objective function

        Returns
        -------
            None

        """
        assert sense in ['min', 'max']
        if sense == 'min':
            mdl.Minimize(xpr)
        else:
            mdl.Maximize(xpr)
        self.obj_expr = xpr

    def get_obj(self, mdl):
        """ Sets get objective direction

        Parameters
        ----------
            mdl : ortools solver
                ortools model

        Returns
        -------
            None

        """
        sense = 'min' if mdl.Objective().minimization() else 'max'
        expr = self.obj_expr if self.obj_expr is not None else 0
        return sense, expr

    def solve(self, mdl, timelimit):
        """ Solves the problem

        Parameters
        -----------
            mdl : ortools solver
                ortools model
            timelimit : int
                time limit in seconds for the solver

        Returns
        -------
            Solution : dict
                A solution if the problem is feasible, the status of the
                of the solver otherwise

        """
        mdl.SetTimeLimit(int(max(1, timelimit*1000)))
        t0 = time.time()
        status = mdl.Solve()
        t1 = time.time()
        stime = t1 - t0
        if status == mdl.OPTIMAL:
            status = 'optimal'
        elif status == mdl.INFEASIBLE:
            status = 'infeasible'
        elif status == mdl.FEASIBLE:
            status = 'solved'
        else:
            status = 'not_solved'
        obj = mdl.Objective().Value()
        var_dict = {k: v.solution_value() for k, v in self.vars.items()}
        return {
            'status': status,
            'obj': obj,
            'time': stime,
            'bound': mdl.Objective().BestBound(),
            'vars': var_dict
        }

    def new_model(self, mdl=None, name=''):
        """ Creates a new model

        Parameters
        ----------
            mdl : ortools solver
                ortools model (default None)
            name : string
                Name of the model (default None)

        Returns
        -------
            Model : ortools solver
                ortools model

        """
        return pywraplp.Solver(name if name else 'milp_model', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    def set_determinism(self, mdl, seed=42):
        # TODO
        pass

    def set_extensive_log(self, mdl):
        # Write the hole model on a txt file
        with open('ortools_model.txt', 'w') as file:
            file.write(mdl.ExportModelAsLpFormat(False).replace('\\', '').replace(',_', ','))
