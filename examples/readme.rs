//! Compile-and-run check for the README "Six lines in" snippet.
//! If this example builds and runs, the README is not fiction.
use rambudget::{BudgetStore, Cost, InsertOutcome};

struct ByteCost;
impl Cost<String, Vec<u8>> for ByteCost {
    fn entry_cost(k: &String, v: &Vec<u8>) -> usize {
        k.len() + v.capacity()
    }
}

fn main() {
    let payload = vec![0u8; 128];
    let store: BudgetStore<String, Vec<u8>, ByteCost> = BudgetStore::new(1 << 30); // 1 GiB cap

    match store.put("session:42".into(), payload.clone()) {
        Ok(InsertOutcome::Stored { charged }) => {
            // admitted: `charged` bytes now reserved
            assert_eq!(charged, "session:42".len() + 128);
        }
        Ok(InsertOutcome::Replaced { freed, charged }) => {
            // old budget refunded, new one gated
            assert!(freed > 0 && charged > 0);
        }
        Err(denied) => {
            // the budget said no — you decide what that means
            let (key, value) = denied.into_pair();
            let _ = (key, value);
        }
    }
    let _ = payload;
}
