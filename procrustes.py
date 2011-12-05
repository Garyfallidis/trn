#!/usr/bin/env python

# this is like script 'procrustes' but it doesn't use the R program

import os, sys

try:
    import numpy
except ImportError:
    sys.stderr.write('\nDownload NumPy from http://www.scipy.org/\n\n')
    sys.exit()

def unquote(s):
    s = s.strip()
    if s[0] == '"' and s[-1] == '"':
        return s[1:-1].replace('\\\\', '\n').replace('\\', '').replace('\n', '\\').strip()
    else:
        return s

def getline(fp, required = False):
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if line and line[0] != '#':
            return unquote(line)

    assert not required
    return False


d, progname = os.path.split(sys.argv[0])

if len(sys.argv) != 4:
    sys.stderr.write('''
Usage: %s target source output

target: input vector file
source: input vector file to be rotated
output: output vector file

source can have less items than target

''' % progname)
    sys.exit()

target = sys.argv[1]
source = sys.argv[2]
output = sys.argv[3]

if os.access(output, os.F_OK):
    sys.stdout.write('\nError %s: file exists: %s\n\n' % (progname, output))
    sys.exit()

labels = []
items = {}

fp = open(source, 'r')
dim = int(getline(fp, True))
while True:
    lbl = getline(fp)
    if not lbl:
        break
    labels.append(lbl)
    items[lbl] = {}
    items[lbl]['source'] = []
    items[lbl]['target'] = []
    for i in range(dim):
        items[lbl]['source'].append(float(getline(fp, True)))
fp.close()

fp = open(target, 'r')
n = int(getline(fp, True))
assert n == dim
while True:
    lbl = getline(fp)
    if not lbl:
        break
    if not items.has_key(lbl):
        items[lbl] = {}
    items[lbl]['target'] = []
    for i in range(dim):
        items[lbl]['target'].append(float(getline(fp, True)))
fp.close()

for lbl in labels:
    assert len(items[lbl]['target']) == dim

X = []
Y = []
for lbl in labels:
    X.append(items[lbl]['target'])
    Y.append(items[lbl]['source'])

################################################################

# # R: package `vegan' version 1.6-10
#
# procrustes <- function (X, Y, scale = TRUE, symmetric = FALSE, scores = "sites", ...) {
#     X <- scores(X, display = scores, ...)
#     Y <- scores(Y, display = scores, ...)
#     if (ncol(X) < ncol(Y)) {
#         warning("X has fewer axes than Y: X adjusted to comform Y\n")
#         addcols <- ncol(Y) - ncol(X)
#         for (i in 1:addcols) X <- cbind(X, 0)
#     }
#     ctrace <- function(MAT) sum(diag(crossprod(MAT)))
#     c <- 1
#     if (symmetric) {
#         X <- scale(X, scale = FALSE)
#         Y <- scale(Y, scale = FALSE)
#         X <- X/sqrt(ctrace(X))
#         Y <- Y/sqrt(ctrace(Y))
#     }
#     xmean <- apply(X, 2, mean)
#     ymean <- apply(Y, 2, mean)
#     if (!symmetric) {
#         X <- scale(X, scale = FALSE)
#         Y <- scale(Y, scale = FALSE)
#     }
#     XY <- crossprod(X, Y)
#     sol <- svd(XY)
#     A <- sol$v %*% t(sol$u)
#     if (scale) {
#         c <- sum(sol$d)/ctrace(Y)
#     }
#     Yrot <- c * Y %*% A
#     b <- xmean - t(A %*% ymean)
#     R2 <- ctrace(X) + c * c * ctrace(Y) - 2 * c * sum(sol$d)
#     reslt <- list(Yrot = Yrot, X = X, ss = R2, rotation = A,
#         translation = b, scale = c, symmetric = symmetric, call = match.call())
#     reslt$svd <- sol
#     class(reslt) <- "procrustes"
#     return(reslt)
# }

def crossprod(x, y = None):
    if y == None:
        y = x
    dimx = len(x[0])
    dimy = len(y[0])
    n = len(x)
    m = [[] for a in range(dimx)]
    for i in range(dimx):
        m[i] = [[] for a in range(dimy)]
    for i in range(dimx):
        for j in range(dimy):
            sum = 0
            for k in range(n):
                sum += x[k][i] * y[k][j]
            m[i][j] = sum
    return m

n = len(X)
for i in range(dim):
    meanx = sum([a[i] for a in X]) / n
    meany = sum([a[i] for a in Y]) / n
    for j in range(n):
        X[j][i] -= meanx
        Y[j][i] -= meany

XY = crossprod(X, Y)

solU, solD, solVt = numpy.linalg.linalg.svd(XY)
solV = numpy.transpose(solVt) # ????

A = crossprod(numpy.transpose(solV), numpy.transpose(solU))

cpY = crossprod(Y)
c = sum(solD) / sum([cpY[i][i] for i in range(dim)])

Yrot = crossprod(numpy.transpose(Y), A)
for i in range(n):
    for j in range(dim):
        Yrot[i][j] *= c

################################################################

fp = open(output, 'w')
fp.write('%i\n' % dim)
for i in range(n):
    fp.write('%s\n' % labels[i])
    for j in range(dim):
        fp.write('%g\n' % Yrot[i][j])
fp.close()
