library(igraph);

rm(list = ls());

# Returns the GCN normalization term as a matrix.
Laplace.Operator = function(grf) {
    Ident = diag(rep(1, length(V(grf))));
    A = grf %>% get.adjacency %>% as.matrix;
    A.hat = A + Ident;
    Dr.hat = diag((grf %>% degree + 1)^(-1/2));
    return(Dr.hat %*% A.hat %*% Dr.hat);
};

MNL = function(X) {
    Z = matrix(data = NA, nrow = nrow(X), ncol = ncol(X));
    for (r in 1:nrow(X)) {
        Z[r,] = exp(X[r,]) / sum(exp(X[r,]));
    }
    return(Z);
}
    

# Example.
grf.test = sample_sbm(n = 120, block.sizes = c(60, 60), directed = F, pref.matrix = rbind(
    c(0.8, 0.6),
    c(0.6, 0.8)
));
plot(grf.test);

x1 = c(rnorm(60, mean = 20, sd = 8), rnorm(60, mean = 30, sd = 8));
x2 = c(rnorm(60, mean = 40, sd = 8), rnorm(60, mean = 30, sd = 8));
x3 = c(rnorm(60, mean = 22, sd = 6), rnorm(60, mean = 25, sd = 4));
X = scale(cbind(x1, x2, x3));

# Consider the case where node 30 and node 90 are known labels. Minimize the loss function.
Y = matrix(rep(0, 120 * 2), nrow = 120, ncol = 2);
Y[1:60,1] = rep(1, 60);
Y[61:120,2] = rep(1, 60);

Loss.Function = function(Z) {
    return(-(Y[30,1] * log(Z[30,1]) + Y[30,2] * log(Z[30,2]) +
        Y[90,1] * log(Z[90,1]) + Y[90,2] * log(Z[90,2])));
};

dw = 1e-6;
lambda = 1;
precision = 1e-3;

# GCN of one layer only.
# Layer 1: MNL. 3 -> 2.
W1.hat = matrix(rnorm(n = 6, mean = 0, sd = 0.01), nrow = 3, ncol = 2);
Z.init = MNL(Laplace.Operator(grf.test) %*% X %*% W1.hat);
plot(Z.init[,1], xlab = "Node ID", ylab = "Z1",
     main = "Performance of untrained GCN, w ~ N(0, 0.01^2)");
GCN.Laplace = Laplace.Operator(grf.test);
while (T) {
    Z = MNL(GCN.Laplace %*% X %*% W1.hat);
    loss = Loss.Function(Z);
    Grad.W1.hat = matrix(data = NA, nrow = nrow(W1.hat), ncol = ncol(W1.hat));
    for (i in 1:nrow(W1.hat)) {
        for (j in 1:ncol(W1.hat)) {
            W1.hat.inc = W1.hat;
            W1.hat.inc[i,j] = W1.hat.inc[i,j] + dw;
            Z.inc = MNL(GCN.Laplace %*% X %*% W1.hat.inc);
            loss.inc = Loss.Function(Z.inc);
            Grad.W1.hat[i,j] = (loss.inc - loss) / dw;
        }
    }
    W1.hat = W1.hat - lambda * Grad.W1.hat;
    print(sum(Grad.W1.hat^2));
    if (sqrt(sum(Grad.W1.hat^2)) < precision) {
        break;
    }
}
plot(Z[,1], xlab = "Node ID", ylab = "Z1", main = "Performance of trained GCN");
lines(rep(0.5, 120), col = "red");

# GCN of two layers.
# Layer 1: Ident. 3 -> 2
# Layer 2: MNL. 2 -> 2
dw = 1e-6;
lambda = 1000;
precision = 1e-5;

W1.hat = matrix(rnorm(6, mean = 2, sd = 0.01), nrow = 3, ncol = 2);
W2.hat = matrix(rnorm(4, mean = 2, sd = 0.01), nrow = 2, ncol = 2);
H1.init = (Laplace.Operator(grf.test) %*% X %*% W1.hat)^3;
Z2.init = MNL(Laplace.Operator(grf.test) %*% H1.init %*% W2.hat);
plot(Z2.init[,1], xlab = "Node ID", ylab = "Z1",
    main = "Performance of untrained GCN, w ~ N(2, 0.01^2)");
GCN.Laplace = Laplace.Operator(grf.test);
while (T) {
    H1 = (GCN.Laplace %*% X %*% W1.hat)^3;
    Z2 = MNL(GCN.Laplace %*% H1 %*% W2.hat);
    loss = Loss.Function(Z2);
    Grad.W1.hat = matrix(data = NA, nrow = nrow(W1.hat), ncol = ncol(W1.hat));
    for (i in 1:nrow(W1.hat)) {
        for (j in 1:ncol(W1.hat)) {
            W1.hat.inc = W1.hat;
            W1.hat.inc[i,j] = W1.hat.inc[i,j] + dw;
            H1.inc = (GCN.Laplace %*% X %*% W1.hat.inc)^3;
            Z2.inc = MNL(GCN.Laplace %*% H1 %*% W2.hat);
            loss.inc = Loss.Function(Z2.inc);
            Grad.W1.hat[i,j] = (loss.inc - loss) / dw;
        }
    }
    Grad.W2.hat = matrix(data = NA, nrow = nrow(W2.hat), ncol = ncol(W2.hat));
    for (i in 1:nrow(W2.hat)) {
        for (j in 1:ncol(W2.hat)) {
            W2.hat.inc = W2.hat;
            W2.hat.inc[i,j] = W2.hat.inc[i,j] + dw;
            H1.inc = (GCN.Laplace %*% X %*% W1.hat)^3;
            Z2.inc = MNL(GCN.Laplace %*% H1 %*% W2.hat.inc);
            loss.inc = Loss.Function(Z2.inc);
            Grad.W2.hat[i,j] = (loss.inc - loss) / dw;
        }
    }
    W1.hat = W1.hat - lambda * Grad.W1.hat;
    W2.hat = W2.hat - lambda * Grad.W2.hat;
    grad.norm = sqrt(sum(Grad.W1.hat^2) + sum(Grad.W2.hat^2));
    print(grad.norm);
    if (grad.norm < precision) {
        break;
    }
}
plot(Z2[,1], xlab = "Node ID", ylab = "Z1", main = "Performance of trained GCN");
lines(rep(0.5, 120), col = "red");
    
