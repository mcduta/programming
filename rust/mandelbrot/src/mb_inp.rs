

pub mod input {
    use std::str::FromStr;
    use num::Complex;


    /// Parse the string `s` as a coordinate pair, like `"400x600"` or `"1.0,0.5"`.
    ///     
    /// Specifically, `s` should have the form <left><sep><right>, where <sep> is
    /// the character given by the `separator` argument, and <left> and <right> are both
    /// strings that can be parsed by `T::from_str`.
    ///
    /// If `s` has the proper form, return `Some<(x, y)>`. If it doesn't parse correctly, return `None`.
    pub fn parse_pair<T: FromStr>(s: &str, separator: char) -> Option<(T, T)> {
        match s.find(separator) {
            None => None,
            Some(index) => {
                match (T::from_str(&s[..index]), T::from_str(&s[index + 1..])) {
                    (Ok(l), Ok(r)) => Some((l, r)),
                    _ => None
                }
            }
        }
    }

    #[test]
    fn test_parse_pair() {
        assert_eq!(parse_pair::<i32>("",        ','), None);
        assert_eq!(parse_pair::<i32>("10,",     ','), None);
        assert_eq!(parse_pair::<i32>(",10",     ','), None);
        assert_eq!(parse_pair::<i32>("10,20",   ','), Some((10, 20)));
        assert_eq!(parse_pair::<i32>("10,20xy", ','), None);
        assert_eq!(parse_pair::<f64>("0.5x",    'x'), None);
        assert_eq!(parse_pair::<f64>("0.5x1.5", 'x'), Some((0.5, 1.5)));
    }




    /// Parse a pair of floating-point numbers separated by a comma as a complex number.
    pub fn parse_complex(s: &str) -> Option<Complex<f64>> {
        match parse_pair(s, ',') {
            Some((re, im)) => Some(Complex { re, im }),
            None => None
        }
    }

    #[test]
    fn test_parse_complex() {
        assert_eq!(parse_complex("1.25,-0.0625"),
                Some(Complex { re: 1.25, im: -0.0625 }));
        assert_eq!(parse_complex(",-0.0625"), None);
    }

}