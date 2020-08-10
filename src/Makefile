CC=gcc
CFLAGS= -std=c99 -O3 -fPIC

SOURCES= disturbing_function.c poisson_series.c nrutil.c fmft.c fmftPy.c
OBJECTS=$(SOURCES:.c=.o)

all: $(SOURCES) libcelmech.so

%.o: %.c
	@echo "Compiling source file $< ..."
	$(CC) -c $(CFLAGS) -o $@ $<

libcelmech.so: $(OBJECTS)
	@echo ""
	@echo "Linking shared library $@ ..."
	$(CC) $(CFLAGS) -shared $(OBJECTS) -o $@

	@echo ""
	@echo "The shared library $< has been created successfully."

clean:
	@echo "Removing object files *.o ..."
	@-rm -f *.o
	@echo "Removing shared library libcelmech.so ..."
	@-rm -f *.so
