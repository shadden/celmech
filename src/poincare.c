
typedef struct PoincareParticle {
	double G,m,mu,M,Lambda,lambda,eta,kappa,rho,sigma;
} PoincareParticle;

struct secular_rk {};
struct secular_splitting {};
struct secular_linear {};

typedef struct simulation {
	double t; // time
	double G; // grav constrant
	double dt;// time step
	int N;
	int allocatedN;

	struct secular_rk ci_secular_rk;
	struct secular_splitting ci_secular_splitting;
	struct secular_linear ci_secular_linear;
	PoincareParticle * particles;

} simulation;

