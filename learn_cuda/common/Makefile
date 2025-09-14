NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80
INCLUDES = -I.

# Target executable
TARGET = test_example

# Source files
SOURCES = test_example.cu

# Build target
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^

# Run the test
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

.PHONY: clean run