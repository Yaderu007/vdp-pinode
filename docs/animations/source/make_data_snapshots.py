# Train MLP run 0 (Kumar setting, 50 windows) exactly like pinode.run_one and record snapshots for the animation.
import sys, json, numpy as np, torch
sys.path.insert(0, "/home/claude/vdp_benchmark/pinode")
import pinode
from torchdiffeq import odeint
torch.set_default_dtype(torch.float64)
eq, sigma, n, run, shoot = "vdp", 0.1, 100, 0, 50
s = dict(pinode.SYSTEMS[eq]); c = s["consts"]
t_np, y_clean, z_np = pinode.make_data(eq, sigma, n, run)
t, z = torch.tensor(t_np), torch.tensor(z_np)
m = n // shoot; h = t_np[1]-t_np[0]; t_loc = torch.arange(m+1)*h; Z = z.reshape(shoot, m, 2)
torch.manual_seed(run)
field = pinode.Field(pinode.make_net("mlp"), pinode.SCALE)
S = torch.nn.Parameter(Z[:, 0, :].clone()); p = torch.nn.Parameter(torch.tensor(float(s["p0"])))
params = list(field.parameters()) + [S, p]
gx = np.linspace(-2.5, 2.5, 13); gv = np.linspace(-3.5, 3.5, 15)
G = torch.tensor([[x, v] for v in gv for x in gx])
tf = torch.linspace(0, 10, 201)
def predict():
    Y = odeint(field, S, t_loc, method="dopri5", rtol=1e-3, atol=1e-3); return Y.permute(1, 0, 2)
def losses():
    Y = predict(); yhat = Y[:, :m, :]
    data = ((yhat - Z) ** 2).mean(); cont = ((Y[:-1, m, :] - S[1:]) ** 2).sum(-1).mean()
    phys = ((field(None, yhat) - pinode.physics(eq, yhat, p, c)) ** 2).mean()
    return data + cont, phys, yhat.reshape(n, 2)
snaps = []
def snap(step, data, phys):
    with torch.no_grad():
        F = field(None, G).numpy()
        Y = odeint(field, torch.tensor([2.0, 0.0]), tf, method="dopri5", rtol=1e-6, atol=1e-6).numpy()  # one free run from the true start
        Yw = predict().numpy()                                     # the 50 shooting windows
    snaps.append(dict(step=step, mu=round(float(p), 4), data=float(data), phys=float(phys),
                      F=np.round(F, 3).tolist(), free=np.round(Y, 3).tolist(), win=np.round(Yw, 3).tolist()))
opt = torch.optim.Adam(params, lr=1e-2)
marks = {0, 25, 50, 100, 200, 300, 500, 700, 1000, 1499}
for it in range(1500):
    opt.zero_grad(); d, ph, _ = losses(); loss = d + ph; loss.backward()
    if it in marks: snap(it, d.item(), ph.item())
    opt.step()
lb = torch.optim.LBFGS(params, lr=1.0, max_iter=300, history_size=50, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
def closure():
    lb.zero_grad(); d_, ph_, _ = losses(); l_ = d_ + ph_; l_.backward(); return l_
lb.step(closure)
d, ph, _ = losses(); snap(1800, d.item(), ph.item())
print("final mu", float(p))
# exact curve, data, and final net weights for a live forward pass
tt = np.linspace(0, 10, 501)
from scipy.integrate import solve_ivp
ex = solve_ivp(pinode.rhs_np("vdp", 1.5, {}), (0, 10), [2.0, 0.0], t_eval=tt, method="DOP853", rtol=1e-10, atol=1e-12).y.T
W = [ [np.round(q.detach().numpy(), 6).tolist() for q in (L.weight, L.bias)] for L in field.net.net if isinstance(L, torch.nn.Linear)]
out = dict(exact=np.round(ex, 4).tolist(), t=np.round(t_np, 4).tolist(), clean=np.round(y_clean, 4).tolist(), data=np.round(z_np, 4).tolist(),
           gx=gx.tolist(), gv=gv.tolist(), snaps=snaps, W=W, scale=pinode.SCALE, nweights=sum(q.numel() for q in field.parameters()))
json.dump(out, open("/tmp/claude-0/-home-claude/997e5c0a-5521-5f74-98d0-ae6fcf7dbcf8/scratchpad/anim/anim_data.json", "w"), separators=(",", ":"))
print([ (s["step"], s["mu"], round(s["data"],4), round(s["phys"],4)) for s in snaps])
