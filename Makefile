r-build:
	@ cd rust; \
	cargo build --release --features="fast print_pred"

r-run:
	@ time -f"%e" 2>data_measured/r-time \
	./rust/target/release/hmm-rust 1>data_measured/r-out

r-build-acc:
	@ cd rust; \
	cargo build --release --features="fast print_acc"

r-run-out:
	./rust/target/release/hmm-rust