import random 
import time
import numpy as np 

KEY_BITS = 16 # or 256, depends on your security level

def int_to_bit_vector(x: int, length: int = KEY_BITS) -> np.ndarray:
    bits = f"{x:0{length}b}"[-length:]
    return np.array([int(b) for b in bits], dtype=np.float32)

def ecc_point_to_bit_vector(P: tuple[int, int], length: int = 32) -> np.ndarray:
    x, y = P
    half = length // 2
    x_bits = int_to_bit_vector(x, half)
    y_bits = int_to_bit_vector(y, half)
    return np.concatenate([x_bits, y_bits])


class EllipticCurve:
    def __init__(self, a, b, p,order=None):
        """Initialize the elliptic curve y^2 = x^3 + ax + b over F_p."""
        self.a = a  
        self.b = b  
        self.p = p  
        if order==None:
            self.order = self.count_points_on_curve()
        else:
            self.order=order 
    
    def is_on_curve(self, x, y):
        """Check if a point (x, y) is on the elliptic curve."""
        return (y**2 - (x**3 + self.a * x + self.b)) % self.p == 0
    
    def add(self, P, Q):
        """Add two points P and Q on the elliptic curve."""
        
        if P is None or P == (None, None):
            return Q
        if Q is None or Q == (None, None):
            return P

        x1, y1 = P
        x2, y2 = Q

        # P + (-P) = O
        if x1 == x2 and (y1 + y2) % self.p == 0:
            return (None, None)

        # Point doubling
        if P == Q:
            # If y1 == 0 → tangent vertical → result is point at infinity
            if y1 % self.p == 0:
                return (None, None)

            s = (3 * x1**2 + self.a) * pow(2 * y1, -1, self.p) % self.p

        # General addition
        else:
            # Safety check (optional but clean)
            if (x2 - x1) % self.p == 0:
                return (None, None)

            s = (y2 - y1) * pow(x2 - x1, -1, self.p) % self.p

        x3 = (s**2 - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p

        return (x3, y3)

    def scalar_mult(self, k, P):
        """Multiply a point P by an integer k using double-and-add algorithm."""
        R = (None, None)
        Q = P

        while k:
            if k & 1:
                R = self.add(R, Q)
            Q = self.add(Q, Q)
            k >>= 1

        return R
    
    def neg(self, point):
        if point is None:
            return None

        x, y = point
        result = x, -y % self.p

        assert self.is_on_curve(result[0],result[1])

        return result
    def legendre_symbol(self, a, p):
        """ Computes the Legendre symbol (a/p) using Euler's criterion """
        return pow(a, (p - 1) // 2, p)

    def sqrt_mod(self, a, p):
        """ Computes the square root of a mod p if it exists """
        if self.legendre_symbol(a, p) != 1:
            return None  
        
        if p % 4 == 3:
            return pow(a, (p + 1) // 4, p)  
        
        raise NotImplementedError("General modular square root needed")
    
    def tonelli_shanks(self,a, p):
        if pow(a, (p - 1) // 2, p) != 1:
            return None  

        if p % 4 == 3:
            return pow(a, (p + 1) // 4, p)

        q = p - 1
        s = 0
        while q % 2 == 0:
            q //= 2
            s += 1

        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1

        c = pow(z, q, p)
        x = pow(a, (q + 1) // 2, p)
        t = pow(a, q, p)
        m = s

        while t != 1:
            i = 1
            temp = pow(t, 2, p)
            while temp != 1:
                temp = pow(temp, 2, p)
                i += 1
                if i == m:
                    return None  

            b = pow(c, 2 ** (m - i - 1), p)
            x = (x * b) % p
            t = (t * b * b) % p
            c = (b * b) % p
            m = i

        return x

    def random_point1(self):
        """ Generate a random point P = (x, y) on the elliptic curve """
        while True:
            x = random.randint(0, self.p - 1) 
            rhs = (x**3 + self.a*x + self.b) % self.p  
            y = self.sqrt_mod(rhs, self.p) 
            if y is not None:
                if random.choice([True, False]):  
                    y = self.p - y  
                return (x, y)
    def random_point(self):
        a=self.a
        b=self.b
        p=self.p
        while True:
            x = random.randint(0, p - 1)
            rhs = (x**3 + a * x + b) % p
            y = self.tonelli_shanks(rhs, p)
            if y is not None:
                if random.choice([True, False]):
                    y = (-y) % p
                return (x, y)
            
    def brute_force_ECDLP(self,P, Q):
        i = 0
        for k in range(self.p-1):
            i = i + 1 
            if self.scalar_mult(k, P) == Q:
                return k , i  
        return None 
    
    def count_points_on_curve(self):
        count = 1  
        for x in range(self.p):
            rhs = (x**3 + self.a * x + self.b) % self.p
            ls = self.legendre_symbol(rhs, self.p)
            
            if ls == 1:
                count += 2  
            elif ls == 0:
                count += 1 

        return count
    
    def point_order(self, P):
        """Returns the order of point P on the given elliptic curve."""
        if P is None:
            return 1  

        Q = P
        order = 1
        while True:
            Q = self.add(Q, P)
            order += 1
            if Q is None:
                return order

    
    def estimate_curve_order(self, trials=5):
        """Estimate the curve order by computing LCM of several point orders."""
        from math import lcm, isqrt
        lower = self.p + 1 - 2 * isqrt(self.p)
        upper = self.p + 1 + 2 * isqrt(self.p)

        candidate_orders = []
        for _ in range(trials):
            P = self.random_point()
            order = self.point_order(P)
            if order:
                candidate_orders.append(order)

        if not candidate_orders:
            return None

        N = candidate_orders[0]
        for o in candidate_orders[1:]:
            N = lcm(N, o)

        if lower <= N <= upper:
            return N
        return None
    def factor(self, n):
        """Naive trial division to factor an integer n."""
        i = 2
        factors = set()
        while i * i <= n:
            if n % i == 0:
                factors.add(i)
                while n % i == 0:
                    n //= i
            i += 1
        if n > 1:
            factors.add(n)
        return list(factors)
    
    def generate_keypair(self, batch_size):
        order = self.order
        size = self.get_key_shape()

        pr_arr = np.empty((batch_size, size[0]))
        pu_arr = np.empty((batch_size, size[1]))

        for i in range(batch_size):
            while True:
                G = self.random_point()
                if G == (None, None):
                    continue  

                private_key = random.randint(1, order - 1)
                public_key = self.scalar_mult(private_key, G)

                if public_key != (None, None):
                    break  

            pr_arr[i] = int_to_bit_vector(private_key)
            pu_arr[i] = ecc_point_to_bit_vector(public_key)

        return pr_arr, pu_arr
    def make_keypair(self,G):
        """Generates a random private-public key pair."""
        private_key = random.randrange(1, curve.order)
        public_key = self.scalar_mult(private_key, G)

        return private_key, public_key

    def get_key_shape(self):
        G = self.random_point()
        order = self.order

        while True:
            private_key = random.randint(1, order - 1)
            public_key = self.scalar_mult(private_key, G)
            if public_key != (None, None):
                break
        
        pr = int_to_bit_vector(private_key)
        pu = ecc_point_to_bit_vector(public_key)
        return len(pr), len(pu)
    
    
        
        
# Example usage:
# Define the elliptic curve y^2 = x^3 + ax + b over F_p
curve = EllipticCurve(
    a=1,
    b=1,
    p=170701,
    order=170701
)
G = curve.random_point()

def main():
    nr_steps = []
    nr_time = []
    P = G
    for i in range (30):
        x = random.randrange(1, curve.order)
        Q = curve.scalar_mult(x, P)
        start_time = time.perf_counter()
        y, steps = curve.brute_force_ECDLP(P, Q)
        end_time = time.perf_counter()
        alg_time = end_time - start_time
        nr_steps.append(steps)
        nr_time.append(alg_time)
        assert x == y
    print('Took', np.mean(nr_steps), 'steps on average')
    print('Took', np.mean(nr_time))


if __name__ == '__main__':
    main()

