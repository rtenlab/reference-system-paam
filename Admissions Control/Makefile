HEADERS =  admissions.h

default: admissions

admissions.o:  admissions.cpp $(HEADERS)
	g++ -c  admissions.cpp -g -o  admissions.o

admissions:  admissions.o
	g++   admissions.o -g -o  admissions

clean:
	-rm -f  admissions.o
	-rm -f  admissions
