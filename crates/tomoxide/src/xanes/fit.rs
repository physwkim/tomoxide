//! Levenberg–Marquardt curve fits used to locate a XANES absorption edge.
//!
//! Ported verbatim (same model, same Jacobians, same solver defaults) from
//! `txm_pal_core::fit` (`~/codes/TXM-Pal-core/src/fit.rs`). Only the unused
//! PyO3-facing helpers (`quadratic_fit` / `gaussian_fit`) were dropped; the two
//! `*_center` entry points the map fitter uses are unchanged.

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DVector, Dyn, Matrix, OVector, Owned, Vector, Vector3, Vector4, U3, U4};

/// `a·x² + b·x + c` least-squares problem.
struct Quadratic {
    params: OVector<f64, U3>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Quadratic {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len());
        Quadratic {
            params: Vector3::<f64>::zeros(),
            x,
            y,
        }
    }
}

impl LeastSquaresProblem<f64, Dyn, U3> for Quadratic {
    type ParameterStorage = Owned<f64, U3>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, U3>;

    fn set_params(&mut self, params: &Vector<f64, U3, Self::ParameterStorage>) {
        self.params = *params;
    }

    fn params(&self) -> Vector<f64, U3, Self::ParameterStorage> {
        self.params
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let residuals = self
            .x
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let temp = self.params[0] * x * x + self.params[1] * x + self.params[2];
                temp - self.y[i]
            })
            .collect::<Vec<_>>();
        Some(DVector::from_vec(residuals))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U3, Self::JacobianStorage>> {
        let jac = self
            .x
            .iter()
            .map(|&x| vec![x * x, x, 1.])
            .collect::<Vec<_>>();
        Some(Matrix::<f64, Dyn, U3, Self::JacobianStorage>::from_row_slice(&jac.concat()))
    }
}

/// Fit a downward parabola and return the vertex energy `-b / 2a`.
pub fn quadratic_fit_center(x: Vec<f64>, y: Vec<f64>, initial_guess: Vec<f64>) -> f64 {
    let mut problem = Quadratic::new(x, y);
    let initial_guess = Vector3::from_vec(initial_guess);
    problem.set_params(&initial_guess);
    let (problem, _) = LevenbergMarquardt::new().minimize(problem);
    problem.params[1] / (-2. * problem.params[0])
}

/// `a·exp(-(x-b)²/2c²) + d` least-squares problem.
struct Gaussian {
    params: OVector<f64, U4>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Gaussian {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len());
        Gaussian {
            params: Vector4::<f64>::zeros(),
            x,
            y,
        }
    }
}

impl LeastSquaresProblem<f64, Dyn, U4> for Gaussian {
    type ParameterStorage = Owned<f64, U4>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, U4>;

    fn set_params(&mut self, params: &Vector<f64, U4, Self::ParameterStorage>) {
        self.params = *params;
    }

    fn params(&self) -> Vector<f64, U4, Self::ParameterStorage> {
        self.params
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let residuals = self
            .x
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let temp = self.params[0]
                    * (-0.5 * ((x - self.params[1]) / self.params[2]).powi(2)).exp()
                    + self.params[3];
                temp - self.y[i]
            })
            .collect::<Vec<_>>();
        Some(DVector::from_vec(residuals))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U4, Self::JacobianStorage>> {
        let a = self.params[0];
        let b = self.params[1];
        let c = self.params[2];

        let jac = self
            .x
            .iter()
            .map(|&x| {
                vec![
                    (-0.5 * ((x - b) / c).powi(2)).exp(),
                    a / c.powi(2) * (x - b) * (-0.5 * ((x - b) / c).powi(2)).exp(),
                    a * (x - b).powi(2) / c.powi(3) * (-0.5 * ((x - b) / c).powi(2)).exp(),
                    1.0,
                ]
            })
            .collect::<Vec<_>>();

        Some(Matrix::<f64, Dyn, U4, Self::JacobianStorage>::from_row_slice(&jac.concat()))
    }
}

/// Fit a Gaussian peak and return its centre energy `b`.
pub fn gaussian_fit_center(x: Vec<f64>, y: Vec<f64>, initial_guess: Vec<f64>) -> f64 {
    let mut problem = Gaussian::new(x, y);
    let initial_guess = Vector4::from_vec(initial_guess);
    problem.set_params(&initial_guess);
    let (problem, _) = LevenbergMarquardt::new().minimize(problem);
    problem.params[1]
}
