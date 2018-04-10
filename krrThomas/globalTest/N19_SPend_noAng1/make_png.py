from ase.io import read, write

a_init = read('init.traj', index=':')
a_MLrelax = read('MLrelax.traj', index=':')

ofname = 'test.png'
write(ofname, a_init[0], show_unit_cell=True)

for i, a in enumerate(a_init):
    ofname = 'train_png/train{}.png'.format(i)
    write(ofname, a, show_unit_cell=True)

for i, a in enumerate(a_MLrelax):
    ofname = 'MLrelax_png/MLrelax{}.png'.format(i)
    write(ofname, a, show_unit_cell=True)

