
from numpy.core.umath_tests import inner1d
import itertools

def makeCovarianceMatrix(num_el, decay, coef=1):
    m = np.eye(num_el)
    for i in range(1,num_el):
        val = [1-decay*i]*(num_el-i)
        m += np.diag(coef*val,+i) + np.diag(coef*val,-i)
    return m

# def generatePDF(x_sample, mu, cov):
#     f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*np.dot(np.dot((x_sample - mu), cov), (x_sample - mu).T))
#     return f

def generatePDF_matrix(x_sample, mu, cov):
    tmp = np.dot((x_sample - mu), cov)
    tmp_T = (x_sample - mu).T
    f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*inner1d(tmp,tmp_T.T))
    return f



# GENERATING POSTERIOR
    # decide how to integrate with rest of the code
    # range of data - within or forwarded????
    # later PRIOR = POSTERIOR
    ########################

mu = np.array([0, 0, 0, 0, 0, 0])
cov = np.eye(6)
st = time.time()

# Initialise to uniform distribution
prior_init = np.ones((len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)))/(np.product([len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)]))
pdf_init = prior_init

# LOOP version
# def updatePDF(pdf, mu, cov):
#     x_range = np.append(np.linspace(-0.5,-0.2,10), np.linspace(0.2,0.5,10))
#     v_range = np.linspace(0.3,1,10)
#     prior_init = np.ones((len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)))/(np.product([len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)]))
#     posterior = np.zeros((len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)))
#     # Update with the gaussian likelihood
#     for idx1, x1 in enumerate(x_range):
#         for idx2, x2 in enumerate(x_range):
#             for idx3, x3 in enumerate(x_range):
#                 for idx4, x4 in enumerate(x_range):
#                     for idx5, x5 in enumerate(v_range):
#                         for idx6, x6 in enumerate(v_range):
#                             posterior[idx1, idx2, idx3, idx4, idx5, idx6] = prior[idx1, idx2, idx3, idx4, idx5, idx6] * generatePDF(np.array([x1,x2,x3,x4,x5,x6]), mu, cov)
#     # Normalise posterior distribution and add it to the previous one
#     posterior = pdf + posterior/np.sum(posterior)
#     # Normalise the final pdf
#     posterior = posterior/np.sum(posterior)
#     return posterior

# PROVERIIIIII ! ! ! ! ! ! !
# MATRIX version
def updatePDF(pdf, mu, cov):
    x_range = np.append(np.linspace(-0.5,-0.2,10), np.linspace(0.2,0.5,10))
    v_range = np.linspace(0.3,1,10)
    prior_init = np.ones((len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)))/(np.product([len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)]))
    posterior = np.zeros((len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)))
    # Update with the gaussian likelihood
    x_samples = np.array([s for s in itertools.product(x_range,x_range,x_range,x_range,v_range,v_range)])
    likelihood = np.reshape(generatePDF_matrix(x_samples, mu, cov), (len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)))
    # Apply Bayes rule
    posterior = prior_init * likelihood
    # Normalise posterior distribution and add it to the previous one
    posterior = pdf + posterior/np.sum(posterior)
    # Normalise the final pdf
    posterior = posterior/np.sum(posterior)
    return posterior


# GETTING SAMPLES (Probability Integral Transform)
    # calculate cdf
    # solve for x, by putting u instead of F(x)
    # u is then the draw from the uniform distribution (which is transformed)
    ########################
def samplePDF(pdf):


    return [idx1, idx2, idx3, idx4, idx5, idx6]






# nsampl = 100    
# mu = np.array([0, 0, 0, 0, 0, 0])
# cov = np.eye(6)
# # cov = makeCovarianceMatrix(2, 0.2)
# x1 = np.linspace(-5,5,num=nsampl)
# x2 = np.linspace(-5,5,num=nsampl)
# x = np.array([x1,x2,x2,x2,x2,x2])
# f = [generatePDF(c, mu, cov) for c in x.T]
# plt.plot(f)
# plt.show()

# mm = np.ones((nsampl,nsampl))
# for i, val_x1 in enumerate(x1):
#     for j, val_x2 in enumerate(x1):
#         mm[i,j]= generatePDF(np.array([val_x1, val_x2]), mu, cov)

# plt.imshow(mm)
# plt.show()