

def makeCovarianceMatrix(num_el, decay):
    m = np.eye(num_el)
    for i in range(1,num_el):
        val = [1-decay*i]*(num_el-i)
        m += np.diag(val,+i) + np.diag(val,-i)
    return m

def generatePDF(x_sample, mu, cov):
    f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*np.dot(np.dot((x_sample - mu).T, cov),(x_sample - mu)))
    return f


# GETTING SAMPLES (Probability Integral Transform)
    # calculate cdf
    # solve for x, by putting u instead of F(x)
    # u is then the draw from the uniform distribution (which is transformed)

def updatePDF(mu):


nsampl = 100    
mu = np.array([0, 0, 0, 0, 0, 0])
cov = np.eye(6)
# cov = makeCovarianceMatrix(2, 0.2)
x1 = np.linspace(-5,5,num=nsampl)
x2 = np.linspace(-5,5,num=nsampl)
x = np.array([x1,x2,x2,x2,x2,x2])
f = [generatePDF(c, mu, cov) for c in x.T]
plt.plot(f)
plt.show()

mm = np.ones((nsampl,nsampl))
for i, val_x1 in enumerate(x1):
    for j, val_x2 in enumerate(x1):
        mm[i,j]= generatePDF(np.array([val_x1, val_x2]), mu, cov)

plt.imshow(mm)
plt.show()
