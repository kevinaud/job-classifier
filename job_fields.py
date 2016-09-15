names = [
	'Management',
	'Business and Financial Operations',
	'Computer and Mathematical',
	'Architecture and Engineering',
	'Life, Physical, and Social Science',
	'Community and Social Services',
	'Legal',
	'Education, Training, and Library',
	'Arts, Design, Entertainment, Sports, and Media',
	'Healthcare Practitioners and Technical',
	'Healthcare Support',
	'Protective Service',
	'Food Preparation and Serving Related',
	'Buildings and Grounds Cleaning and Maintenance',
	'Personal Care and Service',
	'Sales and Related',
	'Office and Administrative Support',
	'Farming, Fishing, and Forestry',
	'Construction and Extraction',
	'Installation, Maintenance, and Repair',
	'Production',
	'Transportation and Material Moving',
	'Military Specific'
]

def shiftIdsDown(IdArray):
	for i in range(IdArray.size):
		IdArray[i] = IdArray[i] - 1

def shiftIdsUp(IdArray):
	for i in range(IdArray.size):
		IdArray[i] = IdArray[i] + 1

