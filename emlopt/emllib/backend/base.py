import inspect

class Backend(object):
    """ CP Solver Wrapper """
    def __init__(self):
        super(Backend, self).__init__()

    def get_lb(self, var):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def get_ub(self, var):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def add_cst(self, mdl, cst, name=None):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def const_eps(self):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def var_cont(self, mdl, lb, ub, name=None):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def var_int(self, mdl, lb, ub, name=None):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def var_bin(self, mdl, name=None):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def xpr_scalprod(self, mdl, coefs, terms):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def xpr_sum(self, mdl, terms):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def xpr_eq(self, mdl, left, right):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def cst_eq(self, mdl, left, right, name=None):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def cst_leq(self, mdl, left, right, name=None):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def cst_geq(self, mdl, left, right, name=None):
        return self.cst_leq(mdl, right, left)

    def cst_indicator(self, mdl, trigger, cst, name=None):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def get_obj(self, mdl):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def set_obj(self, mdl, sense, xpr):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def solve(self, mdl, timelimit):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def new_model(self, name=None):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    # def update_lb(self, bkd, mdl, ml, lb):
    #     raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    # def update_ub(self, bkd, mdl, ml, ub):
    #     raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def set_determinism(self, mdl, seed=42):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')

    def set_extensive_log(self, mdl):
        raise NotImplementedError(f'This method should be implemented in subclasses: {inspect.stack()[0][3]}')
