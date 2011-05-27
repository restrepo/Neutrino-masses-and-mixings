#!/usr/bin/env python
'''Check su(5) texture. The texture is the one given
by the coefficient calculation, e.g:
    y=np.array([1,15,1350,1])/epsfac
    r=np.array([1,3,36,405])/epsfac

and is multiplied by a coefficient matrix obtained by
two different methods:
1) minumum=True
   The coefficient are varied in a wide range and for each one
   the closest local minimum is calculated with fmin_powell.
   At the end the best minimum is chosen.
2) minimum=False
   The coefficient are 25% percent variations about the original
   one and the best minimum is chosen randomly'''
import numpy as np
import scipy.optimize 
import sys
global n,epsilon
epsilon=0.037
n=3

def order(m,V):
    "order eigenvectors in ascending eigenvales order"
    m2=np.array([abs(x)**2 for x in m])
    nu=np.argsort(m2)
    X=[np.reshape(V[:,i],(n,1)) for i in nu]
    return np.hstack(X)

def lepton(eps=0.037):
    alpha=1./np.sqrt(60.)
    eps=0.037
    rdm=np.random.uniform(0.8,1.2,(3,3))
    rdm[1,0]=0. #np.random.uniform(0.1,0.2) #small U_13
    #rdm[0,0]=0.;rdm[0,1]=0.;rdm[0,2]=0. #does not change anything
    #rdm[2,1]=np.random.uniform(0.1,0.5) #help with U_13
    coeff=np.array([[0,0,0],\
                   [-1350.*rdm[1,0]*alpha**3,225.*rdm[1,1]*alpha**2,4724.*rdm[1,2]*alpha**3],\
                   [-3528.*rdm[2,0]*alpha**3,552.*rdm[2,1]*alpha**2,18.*rdm[2,2]*alpha]])


    coeffp=20925.*alpha**3
    l=np.array([[rdm[0,0]*eps**4,rdm[0,1]*eps**5,rdm[0,2]*eps**4],\
               [coeff[1,0]*eps**3,coeff[1,1]*eps**2+coeffp*eps**3,coeff[1,2]*eps**3],[coeff[2,0]*eps**3,coeff[2,1]*eps**2,coeff[2,2]*eps]])

    (m,V)=np.linalg.eig(np.dot(np.transpose(l),l))
#    return order(m,V)
    return np.identity(3)


def chginput(sinput,ry1,ry2,ry3,rr1,rr2,rr3,rx0min,rx0max,\
             rlmin,rlmax,rifin):
    if sinput=='yi':
        ry1=float(raw_input('y1='))
        ry2=float(raw_input('y2='))
        ry3=float(raw_input('y3='))
    if sinput=='ri':
        rr1=float(raw_input('r1='))
        rr2=float(raw_input('r2='))
        rr3=float(raw_input('r3='))
    if sinput=='lambda':
        rlmin=float(raw_input('lmin='))
        rlmax=float(raw_input('lmax='))
    if sinput=='ifin':
        rifin=int(raw_input('ifin='))

    return ry1,ry2,ry3,rr1,rr2,rr3,rx0min,rx0max,\
             rlmin,rlmax,rifin
    
def matrixmdl(x,mdl=3):
    '''define the specific plain matrix model as in
       chrgd_mjrn3_det0.nb
    Input: x -> array
           normalized to maximum entry as in the paper
         : mdl-> integer  
           matrix model as in chrgd_mjrn3_det0.nb
    Output: array (n,n)
            plain matrix model
    Models implemented:
    y=-6: mdl=1 to mdl=4
    y=-8: mdl=5
    y=-3: mdl=6 
    '''
    if mdl==1:
        #y3 is a free parameter
        y1=x[0]; y3=x[1]; r1=x[2]; r2=x[3]
        return np.array([[0.,0.       ,y1*y3],\
                      [0.   ,1./r1    ,-r2*y3/r1],\
                      [y1*y3,-r2*y3/r1,r2**2*y3**2/r1]])

    if mdl==2:
        #y3 is a free parameter
        y1=x[0]; y2=x[1]; r1=x[2]; r2=x[3]
        return np.array([[0.,0.    ,y1/r2],\
                      [0.   ,1./r1 ,-y1/r2],\
                      [y1/r2,-y1/r2,r1*y1**2/r2**2]])

    if mdl==3:
        y1=x[0]; y2=x[1]; r1=x[2]; r2=x[3]; r3=x[4]
        return np.array([[0.,0.            ,y1**2/r3],\
               [0.          ,1./r1         ,-y1*r2/(r1*r3)],\
               [y1**2/r3    ,-y1*r2/(r1*r3),y1**2*r2**2/(r1*r3**2)]])

    if mdl==4:
        y1=x[0]; y2=x[1]; r1=x[2]; r2=x[3]; r3=x[4]
        return np.array([[0.,0.       ,y1*y2],\
                      [0.   ,1./r1    ,-r2*y2/r1],\
                      [y1*y2,-r2*y2/r1,r2**2*y2**2/r1]])

    if mdl==5:
        #y=-8 model
        y1=x[0]; y2=x[1]; r1=x[2]; r2=x[3]; r3=x[4]
        return np.array([[1./r3    ,-r1*y2/r3**2,-r1*y2/r3**2],\
                      [-r1*y2/r3**2,r1**2*y2**2/r3**3,r1**2*y2**2/r3**3],\
                      [-r1*y2/r3**2,r1**2*y2**2/r3**3,r1**2*y2**2/r3**3]])

    if mdl==6:
        y1=x[0]; y2=x[1]; r1=x[2]; r2=x[3]; r3=x[4]
        return np.array([[r1*y1**2/r2**2,r1*y1**2/r2**2,y1/r2],\
                         [r1*y1**2/r2**2,r1*y1**2/r2**2,y1/r2],\
                         [y1/r2         ,y1/r2         ,0]])


def coeffmatrix(mdl,x1,x):
    """Final matrix obtained after the coefficient by coefficient
    multiplication of the coefficient matrix by factor change in x
    input: mdl -> integer
           matrix model  (see matrixmdl)
         : x1 -> array
           Input for neutrino mass matrix 
         : x -> array
             factor change (around 1) for neutrino mass matrix
    """
    v=175.; tanbeta=1; eps=epsilon; Lambda=2.2E+16
    alphaU=0.040
    gU=np.sqrt(4.*np.pi*alphaU)
    V=Lambda/(np.sqrt(5./12.)*gU)
    #m3=v**2*np.sin(np.arctan(tanbeta))**2/(eps*Lambda)*1E9 #eV
    m3=v**2*np.sin(np.arctan(tanbeta))**2/(V)*1E9 #eV
    coeff=matrixmdl(x1,mdl)
    coeffmax=np.abs(coeff).max()
    coeff=coeff/coeffmax
    x2=x[:-1] #x[-1] is the global factor
    return coeffmax*m3*x[-1]**2*matrixmdl(x2,mdl)*coeff
    

def matrixUm(x,mdl=3,y1=17,y2=1350,y3=13500,\
             r1=-10/3.,r2=-50/3.,r3=-250/3.):
    '''Original matrix model (see matrixmdl) multiplied by the
    coefficient matrix to check minimum.
    input: x -> array
             x[:-1] are the normalized coefficients for matrixmdl
             x[-1] is the global factor
         : mdl-> integer  
           matrix model  (see matrixmdl)
    Output: array (n,n)
            Full matrix with final coefficients

    currently, only model 3 is implemented in this way        
    mdl=5 is the general model with two xeroes of texture'''
    epsfac=np.array([np.sqrt(60)**n for n in range(4)])
    y=np.array([1,y1,y2,y3])/epsfac
#    r=np.array([1,-10./3.,-50./3.,-250./3.])/epsfac
#    r=np.array([1,-3.,-36.,-405])/epsfac
    r=np.array([1,r1,r2,r3])/epsfac

    x=np.asarray(x)
    if mdl==1:
        x1=[y[1],y[3],r[1],r[2],r[3]]
        A=coeffmatrix(mdl,x1,x)

    if mdl==2:
        x[1]=1.
        x[4]=1.
        x1=[y[1],y[2],r[1],r[2],r[3]]
        A=coeffmatrix(mdl,x1,x)
        
    if mdl==3:
        x[1]=1.
        #comment to use analysis.py
        #x[5]=0.55
        x1=[y[1],y[2],r[1],r[2],r[3]]
        A=coeffmatrix(mdl,x1,x)

    if mdl==4:
        x[4]=1.
        x1=[y[1],y[2],r[1],r[2],r[3]]
        A=coeffmatrix(mdl,x1,x)

    if mdl==5:
        x1=[y[1],y[2],r[1],r[2],r[3]]
        A=coeffmatrix(mdl,x1,x)

    if mdl==6:
        x[1]=1
        x[4]=1
        x1=[y[1],y[2],r[1],r[2],r[3]]
        A=coeffmatrix(mdl,x1,x)

    if mdl==7:
        A=0.03*x[6]*np.array([[x[0],x[1],x[2]],[x[1],x[3],x[4]],[x[2],x[4],x[5]]])

    if mdl==8:
        x[0]=0.
        A=0.03*x[6]*np.array([[x[0],x[1],x[2]],[x[1],x[3],x[4]],[x[2],x[4],x[5]]])

    if mdl==9:
        a=x[0];b=x[1];c=x[2];d=x[3]
        #[  1.45727054  11.8372313   -9.11060125   1.05323428   0.07920808]
        A=0.03*x[4]*np.array([[a,d,d],[d,b,c],[d,c,b]])
    return A

#A=np.array([[0.8,0.4],[0.4,0.4]])
def oscilation(A):
    '''Check neutrino oscillation data for matrix defined
    in matrixUm.
    input: A array (n,n)
           the output of matrixUm.
    Output: tuple (5)
            neutrino mass matrix properties       
    '''
    (m,V)=np.linalg.eig(A)
    m2=np.array([abs(x)**2 for x in m])
    nu=np.argsort(m2)
    X=[np.reshape(V[:,i],(n,1)) for i in nu]
    U=np.hstack(X)
    Ul=lepton(0.037)
    #Ul=np.identity(3)
    U=np.dot(U,np.transpose(Ul))
    Delta2m32=m2[nu[2]]-m2[nu[1]]
    Delta2m21=m2[nu[1]]-m2[nu[0]]
    t_th12=np.abs(U[0,1]/U[0,0])
    s212=np.sin(np.arctan(t_th12))**2
    t_th23=abs(U[1,2]/U[2,2])
    s223=np.sin(np.arctan(t_th23))**2
    U13=U[0,2]
    return Delta2m32,Delta2m21,s223,s212,U13

def matrix(x,mdl=1,y1=15,y2=1350,y3=13500,\
           r1=-10/3.,r2=-50/3.,r3=-250/3.):
    '''Function to be optimized
    input: x -> array
             x[:-1] are the normalized coefficients for matrixmdl
             x[-1] is the global factor
         : mdl-> integer  
           matrix model  (see matrixmdl)
    Output: float
            value of the function to be minimized
    mdl=7 is the general model'''
        
    s212r=0.304;s223r=0.5
    U213=0.001
    m22=2.40E-3;m21=7.65E-5

    A=matrixUm(x,mdl,y1,y2)
    (Delta2m32,Delta2m21,s223,s212,U13)=oscilation(A)
    return np.abs(s212-s212r)/s212r+np.abs(s223-s223r)/s223r+np.abs(Delta2m32-m22)/m22+np.abs(Delta2m21-m21)/m21+np.abs(U13**2)#-U213)/U213



def check_matrix(A):
    '''Print neutrino parameters for input matrix
    input: A array (n,n)
           the output of matrixUm.
    '''
    dm21r=[7.05E-5,8.34E-5];dm23r=[2.07E-3,2.75E-3]
    s223r=[0.36,0.67];s212r=[0.25,0.37];U213=0.056
    (Delta2m32,Delta2m21,s223,s212,U13)=oscilation(A)
    print ' %.2E < Delta m_{32}^2_{exp} < %.2E' %(dm23r[0],dm23r[1])
    print 'Delta m^2_{23}=%.2E' %(Delta2m32)
    print ' %.2E < Delta m_{21}^2_{exp} < %.2E' %(dm21r[0],dm21r[1])
    print 'Delta m^2_{21}=%.2E' %(Delta2m21)
    print 'tan(theta23)=U[1,2]/U[2,2]=>sin^2(theta23)=%.2f' %s223
    print ' %.2f < tan(theta23)_{exp} < %.2f' %(s223r[0],s223r[1])
    print 'tan(theta12)=U[0,1]/U[0,0]=>sin^2(theta12)=%.2f' %s212
    print ' %.2f < sin^2(theta12)_{exp} < %.2f' %(s212r[0],s212r[1])
    print 'U_13^2=%.3f' %(abs(U13**2))
    print 'U_13_{exp}<%.3f' %U213

def optloop(mdl=1,ifin=10,minimum='False',y1=15,y2=1350,y3=13500,\
            r1=-10/3.,r2=-50/3.,r3=-250/3.,\
            ix0min=0.75,ix0max=1.25,lmin=0.4,lmax=1.2):
    '''Loop to search for the minimum as defined in
         matrix.
    Input:  mdl -> integer  
           matrix model  (see matrixmdl)
         : ifin -> integer
           number of iterations to choose the nimimum
         : minimum -> logical
           Search minimum with fmin_powell (True) or randomly (25%)
         : y1,y2,y3,r1,r2,r3 -> float
           cofficient without sqrt(60) factors (directly from table)
         : ix0min,ix0max -> float
           range of variation for x0
         : lmin,lmax -> float
           range of variation for lambda
    Output: x -> array with found minimum
             x[:-1] are the normalized coefficients
             x[-1] is the global factor
    Fix x0 range below
    Note that x[-1] have one independent range: X0m3
    '''
    if mdl==1 or mdl==2 or mdl==3 or mdl==4 or mdl==6: npar=6
    if mdl==5: npar=6
    if mdl==9: npar=5
    if mdl==7 or mdl==8: npar=7
    if minimum:
        x0min=0.;x0max=2
    else:
        x0min=ix0min;x0max=ix0max
        #x0min=0.7;x0max=1.3
        
    nseed=1
    if nseed==1:
        np.random.seed(nseed)
    X0=np.random.uniform(x0min,x0max,(ifin,npar-1))
    if nseed==1:
        np.random.seed(nseed)
    X0m3=np.random.uniform(lmin,lmax,(ifin,1)) 
    X0=np.hstack([X0,X0m3])
    if minimum:
        argfmin=np.array([scipy.optimize.fmin_powell(matrix,x0,args=(mdl,y1,y2),\
                xtol=1E-14,ftol=1E-14,full_output=1)[1] for x0 in X0]).argmin()
        xmin=scipy.optimize.fmin_powell(matrix,X0[argfmin],args=(mdl,y1,y2),\
                                        xtol=1E-14,ftol=1E-14)
    else:
        argfmin=np.array([matrix(x0,mdl,y1,y2) for x0 in X0]).argmin()
        xmin=X0[argfmin]    

    return xmin

if __name__ == "__main__":
    '''Search for minimum solution.
     y_i=yi/sqrt(60)^i
     r_i=ri/sqrt(60)^i
     Fix yi and ri below  
     Fix range of x0 below
     x0[-1] controls the scale and is fixed
     independently below
    '''
    mdl=3
#    mdlnew=raw_input("Choose the model 1 to 5, (D=3 best model):")
#    if mdlnew: mdl=int(mdlnew)
    print 'mdl:',mdl        
    y1=-8 #-5
    y2=-1924
    y3=13500
    r1=-10/3.
    r2=-50/3.
    r3=-250/3.
    #range of x0:
    x0min=0.75;x0max=1.25
    #range of lambda:
    #lmin=0.4;lmax=10.
    lmin=3.;lmax=10.
    #number of random iterations
    iifin=100000
    #change input parameters
    if len(sys.argv)==2:
        sinput=sys.argv[1]
        (y1,y2,y3,r1,r2,r3,x0min,x0max,lmin,lmax,iifin)=\
            chginput(sinput,y1,y2,y3,r1,r2,r3,x0min,x0max,\
               lmin,lmax,iifin)

    if mdl <1 or mdl > 10: sys.exit()
    minimum=False
    #To search for the realminimum
    if minimum:
        ifin=100
    else:
        ifin=iifin

    print ifin
    x0=optloop(mdl,ifin,minimum,y1,y2,y3,r1,r2,r3,x0min,x0max,\
               lmin,lmax)  #10000000
    print "Model ->",mdl
    print "Check function value:",matrix(x0,mdl,y1,y2)
    namey2='y2'
    if mdl==1: namey2='y3'
    print "x0= factor of [y1,%s,r1,r2,r3,lambda]:" %(namey2)
    print "x0=",x0
    A=matrixUm(x0,mdl,y1,y2)
    print 'normalized matrix:'
    print A/abs(A).max()
    check_matrix(A)
    x1=scipy.optimize.fmin_powell(matrix,x0,args=(mdl,y1,y2),xtol=1E-14,ftol=1E-14)
    print x1

        
        

            
        

