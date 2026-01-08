import os

N = [33, 65, 129, 257, 513, 1025, 2049]
alpha_mach = [(0.0, 0.1), (1.5, 0.1), (0.0, 0.5), (1.5, 0.5), (1.5, 0.75), (0.0, 0.8)]
alpha_mach_to_str = {
    (0.0, 0.1): 'alpha_0-0_mach_0-1',
    (1.5, 0.1): 'alpha_1-5_mach_0-1',
    (1.5, 0.75): 'alpha_1-5_mach_0-75',
    (0.0, 0.8): 'alpha_0-0_mach_0-8',
    (0.0, 0.5): 'alpha_0-0_mach_0-5',
    (1.5, 0.5): 'alpha_1-5_mach_0-5',
}

for n in N :
    for alpha, mach in alpha_mach :
        alpha_mach_str = alpha_mach_to_str[(alpha, mach)]
        filename = f'input_file_fps_{n}_{alpha_mach_str}.txt'
        output_filename = f'output_fps_{n}_{alpha_mach_str}.cgns'
        with open('template_input_fps.txt', 'r') as f :
            content = f.read()
        content = content.replace('GRID_FILENAME=', 
                                  f'GRID_FILENAME=/home/hitra/fullPotentialSolver/pre/naca0012_{n}x{n}.cgns')
        content = content.replace('ALPHA=', f'ALPHA={alpha}')
        content = content.replace('MACH=', f'MACH={mach}')
        content = content.replace('OUTPUT_FILENAME=output/output.cgns', 
                                  f'OUTPUT_FILENAME=/scratch/hitra/output_scale_res/{output_filename}')
        with open(filename, 'w') as f :
            f.write(content)

for alpha, mach in alpha_mach :
    alpha_mach_str = alpha_mach_to_str[(alpha, mach)]
    filename = f'run_fps_{alpha_mach_str}.sh'
    with open('template_run_sbatch.sh', 'r') as f :
            content = f.read()
    content = content.replace('test_job', alpha_mach_str)
    for n in N[:-1] :
        content = content.replace(f'/input_{n}.txt', 
                                f'/home/hitra/fullPotentialSolver/input_file/input_file_fps_{n}_{alpha_mach_str}.txt')
        content = content.replace(f'/output_{n}.txt', 
                                f'/output_scale_res/output_scale_res_fps_{n}_{alpha_mach_str}.txt')
    
    with open(filename, 'w') as f :
        f.write(content)

for alpha, mach in alpha_mach :
    alpha_mach_str = alpha_mach_to_str[(alpha, mach)]
    filename = f'run_fps_{alpha_mach_str}_2049.sh'
    with open('template_run_sbatch_2049.sh', 'r') as f :
            content = f.read()
    content = content.replace('test_job', alpha_mach_str)
    for n in [N[-1]] :
        content = content.replace(f'/input_{n}.txt', 
                                f'/home/hitra/fullPotentialSolver/input_file/input_file_fps_{n}_{alpha_mach_str}.txt')
        content = content.replace(f'/output_{n}.txt', 
                                f'/output_scale_res/output_scale_res_fps_{n}_{alpha_mach_str}.txt')
    
    with open(filename, 'w') as f :
        f.write(content)

# ======================= Cylinder cases =======================
for n in N :
    for alpha, mach in [(0.0, 0.1)] :
        alpha_mach_str = alpha_mach_to_str[(alpha, mach)]
        filename = f'input_file_fps_{n-1}_{alpha_mach_str}_cylinder.txt'
        output_filename = f'output_fps_{n-1}_{alpha_mach_str}_cylinder.cgns'
        with open('template_input_fps.txt', 'r') as f :
            content = f.read()
        content = content.replace('GRID_FILENAME=', 
                                  f'GRID_FILENAME=/home/hitra/fullPotentialSolver/pre/cylinder_{n-1}x{n-1}.cgns')
        content = content.replace('ALPHA=', f'ALPHA={alpha}')
        content = content.replace('MACH=', f'MACH={mach}')
        content = content.replace('OUTPUT_FILENAME=output/output.cgns', 
                                  f'OUTPUT_FILENAME=/scratch/hitra/output_scale_res/{output_filename}')
        with open(filename, 'w') as f :
            f.write(content)

for alpha, mach in [(0.0, 0.1)] :
    alpha_mach_str = alpha_mach_to_str[(alpha, mach)]
    filename = f'run_fps_{alpha_mach_str}_cylinder.sh'
    with open('template_run_sbatch.sh', 'r') as f :
            content = f.read()
    content = content.replace('test_job', alpha_mach_str)
    for n in N[:-1] :
        content = content.replace(f'/input_{n}.txt', 
                                f'/home/hitra/fullPotentialSolver/input_file/input_file_fps_{n-1}_{alpha_mach_str}_cylinder.txt')
        content = content.replace(f'/output_{n}.txt', 
                                f'/output_scale_res/output_scale_res_fps_{n-1}_{alpha_mach_str}_cylinder.txt')
    
    with open(filename, 'w') as f :
        f.write(content)

for alpha, mach in [(0.0, 0.1)] :
    alpha_mach_str = alpha_mach_to_str[(alpha, mach)]
    filename = f'run_fps_{alpha_mach_str}_2048_cylinder.sh'
    with open('template_run_sbatch_2049.sh', 'r') as f :
            content = f.read()
    content = content.replace('test_job', alpha_mach_str)
    for n in [N[-1]] :
        content = content.replace(f'/input_{n}.txt', 
                                f'/home/hitra/fullPotentialSolver/input_file/input_file_fps_{n-1}_{alpha_mach_str}_cylinder.txt')
        content = content.replace(f'/output_{n}.txt', 
                                f'/output_scale_res/output_scale_res_fps_{n-1}_{alpha_mach_str}_cylinder.txt')
    
    with open(filename, 'w') as f :
        f.write(content)