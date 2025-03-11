from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('removal', views.removal, name='removal'),
    path('tutorial', views.tutorial, name='tutorial'),
    path('api/process_image/', views.process_image_api, name='process_image_api'),
    path('autoseg', views.autoseg, name='autoseg'),
    path('api/segment/', views.process_segment_api, name='process_segment_api'),
    path('api/segment_remove/', views.segment_remove, name='segment_remove'),
    path('detail_remove', views.detail_remove, name='detail_remove'),
    path('api/detail_remove/', views.detail_remove_process, name='deatail_remove_api')

    
]
