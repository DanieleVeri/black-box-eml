from . import process, describe
from .. import util


def encode(bkd, net, mdl, net_in, net_out, name):
    """ Encodes the network in the optimization model.

    Codifies each neuron as a variable in the combinatorial problem,
    while each edge is considered as a constraint on the the neurons
    connected.

    Parameters
    ----------
        bkd : :obj:`eml.backend.cplex_backend.CplexBackend`
            Backend Cplex
        net : :obj:`eml.net.describe.DNRNet`
            Network to embed
        mdl : :obj:`docplex.mp.model.Model`
            Model CPLEX
        net_in : list(:obj:`docplex.mp.linear.Var`)
            Input continuous varibles
        net_out : :obj:`docplex.mp.linear.Var`
            Output continuous varibles
        name : string
            Name of the network

    Returns
    -------
        Descriptor : :obj:`eml.util.ModelDesc`
            Descriptor of the neural network

    """
    # Scalar to vector output
    try:
        len(net_out)
    except:
        net_out = [net_out]
    # Build a model descriptor
    desc = util.ModelDesc(net, mdl, name)
    # Process the network layer by layer
    for k, layer in enumerate(net.layers()):
        # Add the layer to the solver wrapper
        for i, neuron in enumerate(layer.neurons()):
            # Add the neuron to the describe
            if k == 0:
                x = net_in[i]
            elif k == net.nlayers()-1:
                x = net_out[i]
            else:
                x = None
            _add_neuron(bkd, desc, neuron, x=x)
    # Enforce basic input bounds
    in_layer = net.layer(0)
    neurons = list(in_layer.neurons())
    for i, var in enumerate(net_in):
        lb = bkd.get_lb(var)
        ub = bkd.get_ub(var)
        neurons[i].update_lb(lb)
        neurons[i].update_ub(ub)
    process.ibr_bounds(net)
    # Return the network descriptor
    return desc


def _add_neuron(bkd, desc, neuron, x=None, is_propagating=False):
    """ Add one neuron to the backend

    Parameters
    ----------
        bkd : :obj:`eml.backend.cplex_backend.CplexBackend`
            Backend CPLEX
        desc : :obj:`eml.util.Modeldesc`
            Model descriptor
        neuron : :obj:`eml.net.describeDNRNeuron`
            Neuron to add
        x : :obj:`docplex.mp.linear.Var`
            Variable representing the neuron (default None)

    Raises
    ------
        ValueError
            If the activation function is not supported

    """
    # Preliminary checks
    if neuron.network() != desc.ml_model():
        raise ValueError('The neuron does not belong to the correct network')
    # if neuron.idx() not in desc._neurons:
    #     desc._neurons.add(neuron.idx())
    # else:
    #     raise ValueError('The neuron has been already added')
    # Obtain network name and inner model
    sn, mdl = desc.name(), desc.model()
    # Obtain current neuron output bounds
    lb, ub = neuron.lb(), neuron.ub()
    # Obtain the neuron index
    idx = neuron.idx()
    # --------------------------------------------------------------------
    # Check whether this is a neuron with an activation function
    # --------------------------------------------------------------------
    net = neuron.network()
    sn = desc.name()
    if not issubclass(neuron.__class__, describe.DNRActNeuron):
        # Build a variable for the model output
        # NOTE this code is duplicated to handle act. dependent bounds
        if x is None:
            x = bkd.var_cont(mdl, lb, ub, '%s_x%s' % (sn, str(idx)))
        desc.store('x', idx, x)
    else:
        # Build an expression for the neuron activation
        coefs, yterms = [1], [neuron.bias()]
        for pidx, wgt in zip(neuron.connected(), neuron.weights()):
            prdx = desc.get('x', pidx)
            coefs.append(wgt)
            yterms.append(prdx)
        y = bkd.xpr_scalprod(mdl, coefs, yterms)
        desc.store('y', idx, y)
        # TODO add the redundant constraints by Sanner
        # TODO add bounding constraints on the y expression
        # ----------------------------------------------------------------
        # Introduce the csts and vars for the activation function
        # ----------------------------------------------------------------
        act = neuron.activation()
        if act == 'relu':
            # Build a variable for the model output
            if x is None:
                x = bkd.var_cont(mdl, max(0, lb), ub, '%s_x%s' % (sn, str(idx)))
            desc.store('x', idx, x)
            # Obtain neuron bounds
            ylb, yub = neuron.ylb(), neuron.yub()
            # Trivial case 1: the neuron is always active
            if ylb >= 0:
                bkd.cst_eq(mdl, x, y, '%s_l%s' % (sn, str(idx)))
            # Trivial case 1: the neuron is always inactive
            elif yub <= 0 and not is_propagating:
                bkd.cst_eq(mdl, x, 0, '%s_z%s' % (sn, str(idx)))
            # Handle the non-trivial case
            else:
                # Enfore the natural bound on the neuron output
                # NOTE if interval based reasoning has been used to
                # # compute bounds, this will be always redundant
                # x.lb = max(0, lb)
                # Introduce a binary activation variable
                z = bkd.var_bin(mdl, '%s_z%s' % (sn, str(idx)))
                desc.store('z', idx, z)
                # Introduce a slack variable
                s = bkd.var_cont(mdl, 0, -ylb, '%s_s%s' % (sn, str(idx)))
                desc.store('s', idx, s)
                # Buid main constraint
                left = bkd.xpr_scalprod(mdl, [1, -1], [x, s])
                bkd.cst_eq(mdl, left, y, '%s_r0%s' % (sn, str(idx)))
                # Build indicator constraints
                right = bkd.xpr_leq(mdl, s, 0)
                bkd.cst_indicator(mdl, z, 1, right, '%s_r1%s' % (sn, str(idx)))
                right = bkd.xpr_leq(mdl, x, 0)
                bkd.cst_indicator(mdl, z, 0, right, '%s_r2%s' % (sn, str(idx)))
        elif act == 'linear':
            # Build a variable for the model output
            if x is None:
                x = bkd.var_cont(mdl, lb, ub, '%s_x%s' % (sn, str(idx)))
            desc.store('x', idx, x)
            # Chain the output variable to the neuron activation
            bkd.cst_eq(mdl, x, y, '%s_l%s' % (sn, str(idx)))
        else:
            raise ValueError('Unsupported "%s" activation function' % act)
