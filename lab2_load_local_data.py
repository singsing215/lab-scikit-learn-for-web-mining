from twenty_newsgroups import load_20newsgroups

# The reference of the data set 20 Newsgroups can be found here:
# http://qwone.com/~jason/20Newsgroups/
newsgroups_data = load_20newsgroups(data_home='./', subset='all')
# list out all the categories name in the dataset
print('newsgroups_data.target_names:')
print(newsgroups_data.target_names)
print('')

print('Size of newsgroups_data.data: %d' % len(newsgroups_data.data))
for i in range(3):
    print('Doc Number %d' % i)
    print('Target Index: %d' % newsgroups_data.target[i])
    print('Doc Type: %s' % newsgroups_data.target_names[newsgroups_data.target[i]])
    print(newsgroups_data.data[i])
    print('')
