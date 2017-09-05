--
-- Created by IntelliJ IDEA.
-- User: lalil0u
-- Date: 19/04/17
-- Time: 16:49
-- To change this template use File | Settings | File Templates.
--

-------UTILS FOR EVE
function std_deviation(alpha, dimension) return alpha/math.sqrt(dimension) end

function REF_transitory(alpha, nevals) return alpha/(2-alpha)*
                                                                (
                                                                1-(1-alpha)^(2*nevals)
                                                                ) end

function norm_l2(vec)
   return math.sqrt(torch.sum(vec:clone():pow(2)))
end

function inplace_columnwise_l2(mat)
    return torch.sum(mat:pow(2), 2):sqrt():resize(mat:size(1))
end

--Updates the relaxed sum in place, using normalized gradients
function update_relaxed_sum(dl_dx, relaxed_sum, alpha)
   local double_dl_dx = dl_dx:clone()
   local n_dl_dx = norm_l2(double_dl_dx)
   local curr_grad =(n_dl_dx>epsilon and double_dl_dx:mul(alpha/n_dl_dx)) or 0
   relaxed_sum:mul((1-alpha))
   relaxed_sum:add(curr_grad)
   return
end

--------------UTILS FOR PH
function init_PH(state)
    state.mu0 = 0
    state.Un = 0
    state.mn = 0
    state.relaxed_loss=0
end

function re_init_after_layer_explosion(state)
    init_PH(state)
    --no re-init

    state.PHCounter = 0
end

function _PH_iter(state)
    --Updating mu0
    state.mu0 = state.PHCounter *state.mu0 + state.relaxed_loss
    state.mu0 = state.mu0/(state.PHCounter+1)

    --Updating Un
    state.Un = state.Un + state.relaxed_loss - state.mu0
    --Updating mn
    if state.Un<state.mn then state.mn = state.Un end

    if state.Un - state.mn>state.lambda then
        --ALERT
        return true
    else
        return false
    end
end

function perform_PH_iter(currLoss, state)

    if state.PHCounter>1/state.mbratio then
        state.relaxed_loss = (1-state.mbratio)*state.relaxed_loss + state.mbratio * currLoss

    else
        state.relaxed_loss = state.PHCounter *state.relaxed_loss + currLoss
        state.relaxed_loss = state.relaxed_loss/(state.PHCounter+1)
    end
    return _PH_iter(state)
end
