## define a variable
#foo=LinNotLearning
#
#
#
#
#excute_in_one_shell:
#	@# multible commands in one shell
#	@# remind me of @echo off
#	export foo='LinLearning';\
#	# use a shell environment variable and a  makefile-owner variable
#	echo Hello, $$foo and not $(foo)
#
#
#excute_in_different_shell:
#	@# multible commands in different shell
#	export bar='LinLearningToo' 
#	echo "Bye, $$bar"

OBJ=warning.md words.md
%.md:%.txt
	mv $^ $@

append_content:$(OBJ)
	echo "LIN COPYRIGHT">$^



