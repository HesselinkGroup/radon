import numpy as np
from radon.fan2para import fan2para, para2fan

def test_fan2para():
    """Test fan2para transformation.
    
    Very simple by-hand verifiable tests.
    """
    r, theta = fan2para(np.pi/4, 0.0, 1.0)
    np.testing.assert_allclose(r, -np.sqrt(2)/2)
    np.testing.assert_allclose(theta, np.pi/4)

    r, theta = fan2para(0.0, -np.pi/4, 1.0)
    np.testing.assert_allclose(r, 0.0, atol=1e-9)
    np.testing.assert_allclose(theta, -np.pi/4)
    
def test_para2fan():
    """Test para2fan transformation.
    
    Very simple by-hand verifiable tests.
    """
    phi, theta_src = para2fan(1.0, 0.0, 1.0)
    np.testing.assert_allclose(phi, -np.pi/2)
    np.testing.assert_allclose(theta_src, np.pi/2, atol=1e-9)
    
    phi, theta_src = para2fan(-1.0, np.pi/4, 1.0)
    np.testing.assert_allclose(phi, np.pi/2)
    np.testing.assert_allclose(theta_src, -np.pi/4, atol=1e-9)
    
def test_fan_para_inversion():
    """Test para2fan and fan2para on list of values.
    
    Test that para2fan inverts fan2para, using some arbitrary fan angles.
    """
    theta_src = np.array([0.0, np.pi/3, 3*np.pi/4])
    phi_fan = np.array([-np.pi/8, 0.0, np.pi/7])

    tt,pp = np.meshgrid(theta_src, phi_fan)
    for t,p in zip(tt.ravel(), pp.ravel()):
        d_src = 2.2 # arbitrary number
        r, theta = fan2para(p, t, d_src)
        p2, t2 = para2fan(r, theta, d_src)

        np.testing.assert_allclose(p, p2, atol=1e-9)
        np.testing.assert_allclose(t, t2, atol=1e-9)
    
test_fan2para()
test_para2fan()
test_fan_para_inversion()
    