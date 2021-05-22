library(igraph);

rm(list = ls());

# Returns the GCN normalization term as a matrix.
GCN.Mat.Norm = function(grf) {
    Ident = diag(rep(1, length(V(grf))));
    A = grf %>% get.adjacency %>% as.matrix;
    A.hat = A + Ident;
    Dr.hat = diag((grf %>% degree + 1)^(-1/2));
    return(Dr.hat %*% A.hat %*% Dr.hat);
};

ReLU = function(x) {
    return(x * as.numeric(x > 0));
}

# Example.
grf.test = sample_sbm(n = 120, block.sizes = c(60, 60), directed = F, pref.matrix = rbind(
    c(0.8, 0.1),
    c(0.1, 0.8)
));
plot(grf.test);

x1 = c(rnorm(60, mean = 20, sd = 4), rnorm(60, mean = 30, sd = 4));
x2 = c(rnorm(60, mean = 40, sd = 4), rnorm(60, mean = 30, sd = 4));
x3 = c(rnorm(60, mean = 22, sd = 3), rnorm(60, mean = 25, sd = 2));

X = cbind(x1, x2, x3);
W0 = matrix(rnorm(6, mean = 1, sd = 1), nrow = 3, ncol = 3);
W1 = matrix(rnorm(6, mean = 1, sd = 1), nrow = 3, ncol = 2);
W2 = rnorm(2, mean = 1, sd = 1);

H0 = X;
H1 = ReLU(GCN.Mat.Norm(grf.test) %*% H0 %*% W0);
H2 = ReLU(GCN.Mat.Norm(grf.test) %*% H1 %*% W1);
H3 = ReLU(GCN.Mat.Norm(grf.test) %*% H2 %*% W2);
H0;
H1;
H2;
H3;
plot(H3, xlab = "Node ID", ylab = "H3", main = "Values of H3 against node, w ~ N(1, 1)");

Best.Cutoff = function(x, label, n.slice) {
    n = length(x);
    x.range = max(x) - min(x);
    h = x.range / n.slice;
    x.cutoff = seq(from = min(x), to = max(x), by = h);
    accuracy = rep(0, length(x.cutoff));
    for (i in 1:length(x.cutoff)) {
        label.hat = as.numeric(x >= x.cutoff[i]);
        label.rev.hat = as.numeric(x < x.cutoff[i]);
        accuracy[i] = max(sum(label.hat == label), sum(label.rev.hat == label)) / n;
    }
    print(sprintf("Max accuracy achievable: %f.", max(accuracy)));
    return(x.cutoff[accuracy == max(accuracy)]);
};

best.h3.cutoff = Best.Cutoff(H3, c(rep(0, 60), rep(1, 60)), n.slice = 100);
best.h3.cutoff;

plot(H3 ~ c(1:120), xlab = "Node ID", ylab = "H3", main = "Values of H3 against node, w ~ N(1, 1)");
for (h3.cutoff in best.h3.cutoff) {
    lines(rep(h3.cutoff, 120) ~ c(1:120), col = "red");
}
