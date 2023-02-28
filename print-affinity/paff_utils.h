# ifndef PAFF_UTILS_H
# define PAFF_UTILS_H

# ifdef __cplusplus
extern "C" {
# endif
int cuda_set_device_id (const int task);
int cuda_get_device_id ();
# ifdef __cplusplus
}
# endif

# endif
