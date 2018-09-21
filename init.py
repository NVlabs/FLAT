from google_drive_downloader import GoogleDriveDownloader  as gdd
import os, json, glob
import pickle
import pdb
import parser
import argparse
import sys

def batch_download(keys, file_dir):
	# download the dataset
	for key in keys:
		folder = '/'.join(key.split('/')[0:-1])+'/'
		if not os.path.exists(folder):
			os.mkdir(folder)
		if not os.path.isfile(key):
			pdb.set_trace()
			gdd.download_file_from_google_drive(
				file_id=file_dir[key]['file_id'],
				dest_path=key,
				unzip=True)

	return 

def main():
	# parsing the input argument
	parser = argparse.ArgumentParser(\
		'determine the part of FLAT dataset to download.'
	)
	parser.add_argument(\
		'-n','--name', 
		metavar='dataset_name',
		type=str, 
		default = 'test',
		nargs = '+',
		help='list the part of FLAT dataset to download'
	)
	parser.add_argument(\
		'-c','--category', 
		metavar='category_name',
		type=str, 
		default = 'kinect',
		help='list the hardware or category of FLAT dataset to download'
	)
	args = parser.parse_args()

	# create the local folder for the dataset
	folder = './FLAT/'
	if not os.path.exists(folder):
		os.mkdir(folder)

	# load the directory list of the flat dataset
	file_dir_name = 'file_dir.pickle'
	with open(file_dir_name, 'rb') as f:
		file_dir = pickle.load(f)

	# inidicate the hardware or trans_render one wants to download	
	flat_flg = args.category
	lists = [key for key in file_dir.keys() if (flat_flg in key)*('.txt' in key)]
	if 'all' in args.name:
		list_flgs = lists
	else:
		list_flgs = []
		for i in range(len(args.name)):
			list_flgs += [key for key in file_dir.keys() \
			if (flat_flg in key)*('/'+args.name[i]+'.txt' in key)]

	# inidicate the hardware or trans_render one wants to download
	flat_flg = 'kinect'

	os.chdir(folder)
	lists = []
	# download the certain list indicated by the flg
	for i in range(len(list_flgs)):
		folder_dir = './'+flat_flg+'/list/'
		filename = folder_dir+list_flgs[i]+'.txt'
		if filename in file_dir.keys():
			batch_download([filename],file_dir)

			# load the file, and read stuffs
			f = open(filename,'r')
			message = f.read()
			files = message.split('\n')
			data_list = files[0:-1]

			# download the images in the list folder
			filename = filename[:-4]+'/'
			keys = [key for key in file_dir.keys() if filename in key]
			batch_download(keys, file_dir)

			# download the files in the datafolder
			keys = [key for key in file_dir.keys() \
				if (key.split('/')[-1] in data_list) \
				and (key.split('/')[1] == flat_flg)
			]
			batch_download(keys, file_dir)


	# download the parameters
	param_id = '1qXvprK-vmS4eJJA4GimjjuqNoNPuAvpw'
	os.chdir('../')
	folder = './params/'
	if not os.path.exists(folder):
		os.mkdir(folder)
	gdd.download_file_from_google_drive(
		file_id=param_id,
		dest_path=folder+'params.zip',
		unzip=True,
	)

	# download the parameters
	param_id = '1gVFmJ4mXkcnjjNHfgQ_BKM4v7woMUYWa'
	os.chdir('./pipe/')
	folder = './models/'
	if not os.path.exists(folder):
		os.mkdir(folder)
	gdd.download_file_from_google_drive(
		file_id=param_id,
		dest_path=folder+'kinect.zip',
		unzip=True
	)

if __name__ == "__main__":
	main()