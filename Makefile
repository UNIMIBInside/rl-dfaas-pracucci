.PHONY = clean

RM_OPTS = --force --verbose --recursive
PYCACHE=$(shell find ./src -name __pycache__)

clean :
	rm $(RM_OPTS) logs
	rm $(RM_OPTS) checkpoints
	rm $(PYCACHE)
