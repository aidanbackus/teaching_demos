use nalgebra::{Matrix3, Vector3};

const STEP_SIZE: f64 = 0.001;
const TOL: f64 = 0.000001;

/// Gradient of f(x) = |Ax + b|^4 + c |x|^2/2, i.e. 4 |Ax + b|^2 * A (Ax + b) + c * x
/// We assume that A is symmetric.
fn grad_f(x: &Vector3<f64>) -> Vector3<f64> {
    let a = Matrix3::new(
        2.0, -1.0, 0.0,
        -1.0, 2.0, -1.0,
        0.0, -1.0, 3.0,
    );
    let b = Vector3::new(1.0, 0.0, -1.0);
    let c = 0.05;

    let y = a * x + b;
    4.0 * y.norm_squared() * (a * y) + c * x
}

fn main() {
    let mut x = Vector3::zeros();
    let mut i = 0;

    loop {
        let g = grad_f(&x);
        if g.norm() < TOL {
            break;
        }
        x -= STEP_SIZE * g;
        i += 1;
    }

    println!("x = {}", x);
    println!("it took {} iterations", i);
}   