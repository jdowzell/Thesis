#!/usr/bin/env python3

import random, string, csv, numpy, math, os, os.path, io, sys

################################################################################

def GenerateListOfFiles():
	"""
	This function searches the files and directories within the current directory
	searching for any file that is in a ".csv" format, and adds makes a note of
	the full pathname, and adds this to a list. This list can then be cycled through
	in a loop, and all of the files can be read.
	"""
	csvList = []
	csv_directories = [x[0] for x in os.walk('.')]

	for csv_root, csv_dirs, csv_files in os.walk('.'):
		for csv_file in csv_files:
			fullpath = os.path.join(csv_root, csv_file)
			if (os.path.splitext(fullpath.lower())[1]).endswith('.csv'):
				csvList.append(fullpath)

	# Now that list of CSV Files has been made, we can
	# work on compiling a list of all exoplanets
	print ("List of CSV files:")
	for i, F in enumerate(csvList):
		print ("{0}: {1}".format(i, F))
	print("\nLength: {}\n".format(len(csvList)))

	return (csvList)

################################################################################

def CheckExitCode(x):
	"""
	Checks to see if a valid exit code is given, and if so, exits the program
	"""

	if(x.lower() == "exit" or x.lower() == "quit" \
	or x.lower() == "q" or x.lower() == "e"):
		print("Exiting program... Goodbye!")
		sys.exit()

################################################################################

def CheckInput(inp,max):
	"""
	This function checks that the given input (inp) ONLY has characters from
	an allowed list, and returns true if it does, or false if it does not.
	Also checks to see if any individual number is less than or equal to the
	maximum number allowed (max).
	* If any single number is > number of headers (or < 0), the program alerts the user.
	* If the codes aren't valid, it returns FALSE
	* If the codes ARE valid, it returns a list of valid header numbers
	"""

	#print("Input was: {}".format(inp))

	# First of all, if the input string is "e", "q", "exit", or "quit"
	# (in any mixture of upper- or lower- case) then just end the program
	CheckExitCode(inp)

	# Filter/split the input into a list of all non-blank entities
	# EG an input of "3,,,5,6" would become [3,5,6], ignoring the empty spaces
	newStr = list(filter(None,inp.split(",")))
	#print(newStr)

	# Additionally, split up the new string to see if any one number is > max
	for c in newStr:

		# Check if all of the values selected are integers;
		# "It's easier to ask for forgiveness than to ask for permission"
		while True:
			try:
				int(c)
				break
			except ValueError:
				#print("EXIT VALUEERROR")
				return(False)
		if ((int(c) >= int(max)) or (int(c) < int(0))):
			#print("EXIT OUTOFBOUNDS")
			return False
	return (newStr)

	# Set up the valid character dictionary
#	validInputDict=dict.fromkeys("0123456789")
#
#	# Check to see if EVERY character in (inp) appears in the allowed list
#	for i in newStr:
#		allOK = all(c in validInputDict for c in i)
#	#print("Character Check: {}".format(c in validInputDict for c in newStr))
#	if(allOK==False):
#		print("EXIT NOTINLIST")
#		return(False)
#	else:
#		return(newStr)

################################################################################

def ReadFile(fl):
	"""
	Stuff
	"""
	with open(fl, 'r') as curDB:
		planetCSV = csv.reader(curDB, delimiter=',', quotechar='"')

		# Skip all lines that begin with a '#' (ie, a comment)
		hLine = False
		for rNum, row in enumerate(planetCSV):
			if(row[0][0]=="#"):
				continue

			# The immediate next line is the header line
			if(hLine == False):
				hLine = row
				print("Headers:")

				for iNum, item in enumerate(hLine):
					print("{}:\t{}".format(iNum,item))
					
				# Set up the input loop flags and variables
				validInputGiven = False
				validHeaders = ""

				# Start the input loop
				while(validInputGiven==False):
					headerList = input("\nPlease type in the header inputs you "
					"want to use.\nTo concatenate multiple headers, please use "
					"a ',' symbol between headers.\n"
					"For example, to use a concatenation of headers 3 and 7, "
					"please type '3,7 (without spaces)'.\n"
					"If at any time you want to stop the program, please type "
					"'exit' or 'quit'.\n")

					# Check if the input is valid
					validHeaders = CheckInput(headerList,len(hLine))
					if(validHeaders):
						validInputGiven = True
					else:
						print("\nThe input '{}' was invalid; please try again.\n".format(headerList))

				# If valid, set the flag to True, so the input loop stops
				validInputGiven = True

				# We have a header line, so set this flag to True also
				hLine = True
				continue
			
			#print(row[0])
			#sys.exit()

			# The rest of the lines are the data
			for i, iLine in enumerate(row):
				print("{0}:\t{1}".format(i, row))

################################################################################

def main():

	#stuff
	ReadFile(GenerateListOfFiles()[1])
	
################################################################################

# OKAY 3 2 1 LET'S JAM
# https://www.youtube.com/watch?v=n2rVnRwW0h8
main()
