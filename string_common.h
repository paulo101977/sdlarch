#include <string.h>

union string_list_elem_attr
{
   bool  b;
   int   i;
   void *p;
};

struct string_list_elem
{
   char *data;
   void *userdata;
   union string_list_elem_attr attr;
};

struct string_list
{
   struct string_list_elem *elems;
   size_t size;
   size_t cap;
};

static bool string_is_equal(const char *a, const char *b)
{
   return (a && b) ? !strcmp(a, b) : false;
}

static bool string_list_deinitialize_internal(struct string_list *list)
{
   if (!list)
      return false;

   if (list->elems)
   {
      unsigned i;
      for (i = 0; i < (unsigned)list->size; i++)
      {
         if (list->elems[i].data)
            free(list->elems[i].data);
         if (list->elems[i].userdata)
            free(list->elems[i].userdata);
         list->elems[i].data     = NULL;
         list->elems[i].userdata = NULL;
      }

      free(list->elems);
   }

   list->elems = NULL;

   return true;
}


void string_list_free(struct string_list *list)
{
   if (!list)
      return;

   string_list_deinitialize_internal(list);

   free(list);
}

struct string_list *string_list_new(void)
{
   struct string_list_elem *
      elems                 = NULL;
   struct string_list *list = (struct string_list*)
      malloc(sizeof(*list));
   if (!list)
      return NULL;

   list->cap                = 0;
   list->size               = 0;
   list->elems              = NULL;

   if (!(elems = (struct string_list_elem*)
      calloc(32, sizeof(*elems))))
   {
      string_list_free(list);
      return NULL;
   }

   list->elems              = elems;
   list->cap                = 32;

   return list;
}

bool string_list_capacity(struct string_list *list, size_t cap)
{
   struct string_list_elem *new_data = (struct string_list_elem*)
      realloc(list->elems, cap * sizeof(*new_data));

   if (!new_data)
      return false;

   if (cap > list->cap)
      memset(&new_data[list->cap], 0, sizeof(*new_data) * (cap - list->cap));

   list->elems = new_data;
   list->cap   = cap;
   return true;
}

bool string_list_append(struct string_list *list, const char *elem,
      union string_list_elem_attr attr)
{
   char *data_dup = NULL;

   /* Note: If 'list' is incorrectly initialised
    * (i.e. if struct is zero initialised and
    * string_list_initialize() is not called on
    * it) capacity will be zero. This will cause
    * a segfault. Handle this case by forcing the new
    * capacity to a fixed size of 32 */
   if (      list->size >= list->cap
         && !string_list_capacity(list,
               (list->cap > 0) ? (list->cap * 2) : 32))
      return false;

   if (!(data_dup = strdup(elem)))
      return false;

   list->elems[list->size].data = data_dup;
   list->elems[list->size].attr = attr;

   list->size++;
   return true;
}