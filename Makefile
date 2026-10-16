.PHONY: install train test cpp

install:
	pip install -e python/
	pip install -r training/requirements.txt

train:
	PYTHONPATH=python/src:training python -m rapidstamp_train.train --out models/demo

test:
	PYTHONPATH=python/src:training python -m pytest -q

cpp:
	cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
	cmake --build cpp/build -j
	ctest --test-dir cpp/build --output-on-failure
